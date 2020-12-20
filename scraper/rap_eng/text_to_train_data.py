# coding=utf-8
import os
import re

directory = os.fsencode("texts/")

def getContent(file):
  with open("texts/" + file, "r") as f:
    return re.sub(r"\[.*\]", "", f.read())

def appendTo(file, text):
  with open("traindata/" + file, "a") as f:
    f.write(text)
    f.write("\n<eor>\n")


for dirAuthor in os.listdir(directory):

  author = os.fsdecode(dirAuthor)

  if author.startswith(".DS_Store"):
    continue

  print(author + "\n---")
  songs = os.listdir(directory + dirAuthor)


  # validation and test samples
  if len(songs) > 2:
    validationSample = os.fsdecode(songs[-2])
    print("  validation sample:  " + validationSample)
    appendTo("valid.txt", getContent(author + "/" + validationSample))

    testSample = os.fsdecode(songs[-1])
    print("  test sample:  " + testSample)
    appendTo("test.txt", getContent(author + "/" + testSample))

    songs = songs[:-2]

  print("  train samples:  ")
  for songF in songs:
    song = os.fsdecode(songF)
    print("    " + os.fsdecode(song))
    appendTo("train.txt", getContent(author + "/" + song))

