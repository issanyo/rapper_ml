# coding=utf-8
import requests
from bs4 import BeautifulSoup
from genius_artist import *
import os
import re
import threading
import queue
import time

# Collect first page of artists’ list
page = requests.get('https://it.wikipedia.org/wiki/Categoria:Rapper_italiani')

# Create a BeautifulSoup object
soup = BeautifulSoup(page.text, 'html.parser')

artist_name_list_items = soup.select(".mw-category ul li a")

start = 123

workQueue = queue.Queue(10)
queueLock = threading.Lock()

class downloadTextThread(threading.Thread):
  def __init__(self, threadID, name, file, songName, songLink, q):
    threading.Thread.__init__(self)
    self.threadID = threadID
    self.name = name
    self.file = file
    self.songName = songName
    self.songLink = songLink
    self.q = q

  def run(self):
    print("  " + self.songName + " - " + self.songLink)
    songText = get_song_lyrics(self.songLink)

    if songText:
      with open(self.file, "w") as text_file:
        text_file.write(songText)

    # update the thread working number
    queueLock.acquire()
    if not workQueue.empty():
      self.q.get()
    queueLock.release()


# Iterate over artists
for key,artist in enumerate(artist_name_list_items[start:]):
  artistName = artist.text

  artistName = re.sub(r"\(.*\)", "", artistName)

  artistName = artistName.strip()

  print(str(key+start) + ": " + artistName + "\n----")

  directory = "texts/" + artistName + "/"

  if not os.path.exists(directory):
    os.makedirs(directory)

  #for songLink,songName in get_songs(artistName):
  #  songName = songName.replace("/", "\\")
    #print("  " + songName + " - " + songLink)
    #songText = get_song_lyrics(songLink)

    #if songText:
    #  with open(directory + songName, "w") as text_file:
    #    text_file.write(songText)

  i = 0
  for songLink, songName in get_songs(artistName):
    songName = songName.replace("/", "\\")

    queueLock.acquire()
    workQueue.put(str(i))
    queueLock.release()

    thread = downloadTextThread(i, "Thread-"+str(i), file=directory + songName, songName=songName, songLink=songLink, q=workQueue)
    thread.start()

    # Wait for queue to leave space
    while workQueue.full():
      time.sleep(1)

    i+=1

# Wait for queue empty
while not workQueue.empty():
  time.sleep(1)
