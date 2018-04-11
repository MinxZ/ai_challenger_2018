import glob
import multiprocessing as mp
import os
import random
import sys
import tarfile
import zipfile
from collections import defaultdict
from io import StringIO

import cv2
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from tqdm import tqdm


def resizeAndPad(img, size, padColor=255):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw:  # shrinking image
        interp = cv2.INTER_AREA
    else:  # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    # if on Python 2, you might need to cast as a float: float(w)/h
    aspect = w / h

    # compute scaling and pad sizing
    if aspect > 1:  # horizontal image
        new_w = sw
        new_h = np.round(new_w / aspect).astype(int)
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(
            int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:  # vertical image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(
            int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:  # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    # color image but only one color provided
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)):
        padColor = [padColor] * 3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(
        scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img


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
label2name = dict(zip(list_train, names_train))

idx = -1
data = []
y = []
for label, name in sorted(label2name.items(), key=lambda x: x[0]):
    idx += 1
    print(f'{idx} {label} {name}')
    dir_path = f'{zl_path}/fruits/{name}'
    width = 299
    image_list = []
    for filename in glob.glob(f'{dir_path}/*.jpg'):  # assuming gif
        image_list.append(filename)
    for i in tqdm(range(len(image_list))):
        image_path = image_list[i]
        s_img = cv2.imread(image_path)
        b, g, r = cv2.split(s_img)       # get b,g,r
        image_np = cv2.merge([r, g, b])     # switch it to rgb
        resize_pad_img = resizeAndPad(image_np, (width, width))
        data.append(resize_pad_img)
        y.append(idx)
data = np.array(data)
y = np_utils.to_categorical(y)
np.save(f'{zl_path}/fruits/X', data)
np.save(f'{zl_path}/fruits/y', y)
