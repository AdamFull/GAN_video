import urllib
from urllib import request
from urllib.parse import quote
import re, os, sys

link = "https://www.youtube.com"
search_link = f"{link}/results?search_query="
video_link = f"{link}/watch?v="

def parse(usr_request):
    output = []
    sq = search_link+usr_request
    doc = urllib.request.urlopen(sq).read().decode('cp1251',errors='ignore')
    match = re.findall("\?v\=(.+?)\"", doc)
    if not (match is None):
        for link in match:
            if len(link)<25:
                output.append(link)
    for i in range(len(output)):
        output[i] = video_link+output[i]
    return output