# coding=utf-8
import requests
from bs4 import BeautifulSoup
import json

def _retreive_artist_id(name):
  content = requests.get("https://genius.com/artists/" + name)

  soup = BeautifulSoup(content.text, 'html.parser')
  
  if len(soup.select('head meta[name="newrelic-resource-path"]')) == 0:
    return None
  return soup.select('head meta[name="newrelic-resource-path"]')[0].get("content")

def get_songs(artist):
  id = _retreive_artist_id(artist)
  if not id:
    return []

  content = requests.get("https://genius.com/api" + id + "/songs?page=1&sort=popularity")
  json_content = json.loads(content.text)
  if 'response' not in json_content:
    return []
  songs = [(song["url"],song["title"]) for song in json_content['response']["songs"]]

  while json_content['response']['next_page']:
    content = requests.get("https://genius.com/api" + id + "/songs?page= " + str(json_content['response']['next_page']) + " &sort=popularity")
    json_content = json.loads(content.text)
    songs.extend([(song["url"],song["title"]) for song in json_content['response']["songs"]])

  return songs

def get_song_lyrics(link):
  content = requests.get(link)

  soup = BeautifulSoup(content.text, 'html.parser')

  if(len(soup.select('div.lyrics p')) > 0):
    return soup.select('div.lyrics p')[0].text
  return None
