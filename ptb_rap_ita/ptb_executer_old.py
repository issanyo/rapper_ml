from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

import reader
import util

from tensorflow.python.client import device_lib

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer("num_gpus", 1,
                     "If larger than 1, Grappler AutoParallel optimizer "
                     "will create multiple training replicas with each GPU "
                     "running one replica.")
flags.DEFINE_string("rnn_mode", None,
                    "The low level implementation of lstm cell: one of CUDNN, "
                    "BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, "
                    "and lstm_block_cell classes.")
FLAGS = flags.FLAGS
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32

class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)

class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_):
    self._is_training = is_training
    self._input = input_
    self._rnn_params = None
    self._cell = None
    self.batch_size = input_.batch_size
    self.num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size
	
    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

    output, state = self._build_rnn_graph_lstm(inputs, config, is_training)

    softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
     # Reshape logits to be a 3-D tensor for sequence loss
    logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

    # Use the contrib sequence loss and average over the batches
    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        input_.targets,
        tf.ones([self.batch_size, self.num_steps], dtype=data_type()),
        average_across_timesteps=False,
        average_across_batch=True)

    # Update the cost
    self._cost = tf.reduce_sum(loss)
    self._final_state = state
    self.probas = tf.nn.softmax(logits, name='probas')
	
  def _get_lstm_cell(self, config, is_training):
    if config.rnn_mode == BASIC:
      return tf.contrib.rnn.BasicLSTMCell(
          config.hidden_size, forget_bias=0.0, state_is_tuple=True,
          reuse=False)
    if config.rnn_mode == BLOCK:
      return print("WTF")
    raise ValueError("rnn_mode %s not supported" % config.rnn_mode)
		
  def _build_rnn_graph(self, inputs, config, is_training):
    if config.rnn_mode == CUDNN:
      return self._build_rnn_graph_cudnn(inputs, config, is_training)
    else:
      return self._build_rnn_graph_lstm(inputs, config, is_training)

  def _build_rnn_graph_lstm(self, inputs, config, is_training):
    """Build the inference graph using canonical LSTM cells."""
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def make_cell():
      cell = self._get_lstm_cell(config, is_training)
      if is_training and config.keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=config.keep_prob)
      return cell

    cell = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(config.batch_size, data_type())
    state = self._initial_state
    # Simplified version of tf.nn.static_rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use tf.nn.static_rnn() or tf.nn.static_state_saving_rnn().
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
    # outputs, state = tf.nn.static_rnn(cell, inputs,
    #                                   initial_state=self._initial_state)
    outputs = []
    with tf.variable_scope("RNN"):
      for time_step in range(self.num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)
    output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
    return output, state

  @property
  def input(self):
    return self._input
  @input.setter
  def input(self, value):
    self._input = value
  @property
  def cost(self):
    return self._cost
  @property
  def final_state(self):
    return self._final_state
  @property
  def initial_state(self):
    return self._initial_state


class YoConfig(object):
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 1
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 6
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 1
  vocab_size = 10000
  rnn_mode = BLOCK

def get_config():
  """Get model config."""
  config = None
  
  if FLAGS.model == "yo":
    config = YoConfig()
  elif FLAGS.model == "small":
    config = SmallConfig()
  elif FLAGS.model == "medium":
    config = MediumConfig()
  elif FLAGS.model == "large":
    config = LargeConfig()
  elif FLAGS.model == "test":
    config = TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)
  if FLAGS.rnn_mode:
    config.rnn_mode = FLAGS.rnn_mode
  if FLAGS.num_gpus != 1 or tf.__version__ < "1.3.0" :
    config.rnn_mode = BASIC
  return config

def sample_from_pmf(probas):
  t = np.cumsum(probas)
  s = np.sum(probas)
  return int(np.searchsorted(t, np.random.rand(1) * s))

def generate_text(session, model, word_to_index, index_to_word, seed='</s>', n_sentences=10):
  sentence_cnt = 0
  input_seeds_id = [word_to_index[w] for w in seed.split()]
  state = session.run(model.initial_state)

  # Initiate network with seeds up to the before last word:
  for x in input_seeds_id[:-1]:
    feed_dict = {model.initial_state: state, model.input.input_data: [[x]]}
    state = session.run([model.final_state], feed_dict)

    text = seed
    # Generate a new sample from previous, starting at last word in seed
    input_id = [[input_seeds_id[-1]]]
    while sentence_cnt < n_sentences:
      feed_dict = {model.input.input_data: input_id, model.initial_state: state}
      probas, state = session.run([model.probas, model.final_state], feed_dict=feed_dict)
      sampled_word = sample_from_pmf(probas[0])
      if sampled_word == word_to_index['<eos>']:
        text += '.\n'
        sentence_cnt += 1
      else:
        text += ' ' + index_to_word[sampled_word]
        input_wordid = [[sampled_word]]

    print(text)


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")
  gpus = [
      x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"
  ]
  if FLAGS.num_gpus > len(gpus):
    raise ValueError(
        "Your machine has only %d gpus "
        "which is less than the requested --num_gpus=%d."
        % (len(gpus), FLAGS.num_gpus))

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, vocabulary, word_to_id, id_to_word = raw_data

  eval_config = get_config()

  initializer = tf.random_uniform_initializer(-eval_config.init_scale, eval_config.init_scale)
	
  sess = tf.Session()

  valid_input = PTBInput(config=eval_config, data=test_data, name=None)
  with tf.variable_scope("Model", reuse=None, initializer=initializer):
    tf.global_variables_initializer()
    mtest = PTBModel(is_training=False, config=eval_config, input_=valid_input)

  sess.run(tf.global_variables_initializer())

  saver = tf.train.Saver()
  saver.restore(sess, tf.train.latest_checkpoint(FLAGS.save_path))

  msg = 'Reading model parameters from %s' % FLAGS.save_path
  print(msg)

  while True:
    print(generate_text(sess, mtest, word_to_id, id_to_word, seed="this boy", n_sentences=1))
    try:
      input('press Enter to continue ...\n')
    except KeyboardInterrupt:
      print('\b\bQuiting now...')
      break

if __name__ == "__main__":
  tf.app.run()
