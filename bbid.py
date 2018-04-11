#!/usr/bin/env python3
import argparse
import hashlib
import imghdr
import os
import pickle
import posixpath
import re
import signal
import socket
import threading
import time
import urllib.parse
import urllib.request


def download(url, output_dir):
    if url in tried_urls:
        return
    pool_sema.acquire()
    path = urllib.parse.urlsplit(url).path
    filename = posixpath.basename(path).split(
        '?')[0]  # Strip GET parameters from filename
    name, ext = os.path.splitext(filename)
    name = name[:36]
    filename = name + ext

    i = 0
    while os.path.exists(os.path.join(output_dir, filename)) or filename in in_progress:
        i += 1
        filename = "%s-%d%s" % (name, i, ext)
    in_progress.append(filename)
    try:
        request = urllib.request.Request(url, None, urlopenheader)
        image = urllib.request.urlopen(request).read()
        if not imghdr.what(None, image):
            print('FAIL: Invalid image, not saving ' + filename)
            return

        md5_key = hashlib.md5(image).hexdigest()
        if md5_key in image_md5s:
            print('FAIL: Image is a duplicate of ' +
                  image_md5s[md5_key] + ', not saving ' + filename)
            return

        image_md5s[md5_key] = filename

        imagefile = open(os.path.join(output_dir, filename), 'wb')
        imagefile.write(image)
        imagefile.close()
        print("OK: " + filename)
        tried_urls.append(url)
    except Exception as e:
        print("FAIL: " + filename)
    finally:
        in_progress.remove(filename)
        pool_sema.release()


def fetch_images_from_keyword(keyword, output_dir, filters, limit):
    current = 0
    last = ''
    while True:
        request_url = 'https://www.bing.com/images/async?q=' + urllib.parse.quote_plus(
            keyword) + '&first=' + str(current) + '&adlt=' + '' + '&qft=' + ('' if filters is None else filters)
        request = urllib.request.Request(
            request_url, None, headers=urlopenheader)
        response = urllib.request.urlopen(request)
        html = response.read().decode('utf8')
        links = re.findall('murl&quot;:&quot;(.*?)&quot;', html)
        try:
            if links[-1] == last:
                return
            for index, link in enumerate(links):
                if limit is not None and current + index >= limit:
                    return
                t = threading.Thread(target=download, args=(link, output_dir))
                t.start()
            last = links[-1]
            current += bingcount
        except IndexError:
            print('No search results for "{0}"'.format(keyword))
            return
        time.sleep(0.01)


def backup_history(*args):
    download_history = open(os.path.join(
        output_dir, 'download_history.pickle'), 'wb')
    pickle.dump(tried_urls, download_history)
    # We are working with the copy, because length of input variable for pickle must not be changed during dumping
    copied_image_md5s = dict(image_md5s)
    pickle.dump(copied_image_md5s, download_history)
    download_history.close()
    print('history_dumped')
    if args:
        exit(0)



# config
zl_path = '/Users/z/zl'
fsplit = open(f'{zl_path}/fruits/fruits_test_label.txt', 'r', encoding='UTF-8')
lines_label = fsplit.readlines()
fsplit.close()
list_train = list()
names_train = list()
for each in lines_label:
    tokens = each.split(', ')
    list_train.append(tokens[0])
    names_train.append(tokens[1])

keyword = 'radish'
output_dir = f'{zl_path}/fruits/{keyword}'  # default output dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
adult_filter = True  # Do not disable adult filter by default
pool_sema = threading.BoundedSemaphore(
    value=20)  # max number of download threads
bingcount = 35  # default bing paging
socket.setdefaulttimeout(2)

in_progress = tried_urls = []
image_md5s = {}
urlopenheader = {
    'User-Agent': 'Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'}

fetch_images_from_keyword(keyword, output_dir, None, 500)
