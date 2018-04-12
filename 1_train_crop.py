from __future__ import absolute_import, division, print_function

import multiprocessing as mp
import random
from collections import defaultdict

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
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


def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text


dic = {'[': '', ']': '', '\n': ''}

# animals_fruits = 'animals'
# animals_fruits = 'fruits'
for animals_fruits in ['animals', 'fruits']:
    zl_path = '/data/zl'
    dir_path = f'{zl_path}/ai_challenger_zsl2018_train_test_a_20180321/zsl_a_{animals_fruits}_train_20180321'
    fname = f'{dir_path}/zsl_a_{animals_fruits}_train_annotations_labels_20180321.txt'
    data_path = f'{dir_path}/zsl_a_{animals_fruits}_train_images_20180321'
    with open(fname) as f:
        content = f.readlines()
    content = [replace_all(x, dic) for x in content]
    content = [x.split(', ') for x in content]

    row = 1
    column = 1
    width = 299
    data = []
    labels = []
    # plt.figure(figsize=(20, 20))
    for x in tqdm(range(len(content))):
        s_img = cv2.imread(data_path + '/' + content[x][6])
        b, g, r = cv2.split(s_img)       # get b,g,r
        rgb_img = cv2.merge([r, g, b])     # switch it to rgb
        x_1 = int(content[x][2])
        x_2 = int(content[x][4])
        y_1 = int(content[x][3])
        y_2 = int(content[x][5])
        crop_img = rgb_img[x_1:x_2, y_1:y_2]
        # crop_img = rgb_img
        resize_pad_img = resizeAndPad(crop_img, (width, width))
        data.append(resize_pad_img)
        # cv2.imwrite(
        #     data_path + '/' + content[x][6], resize_pad_img)
        # plt.subplot(row, column, x + 1)
        # plt.imshow(crop_img)
        labels.append(content[x][1])
    data = np.array(data)
    np.save(f'{zl_path}/{animals_fruits}/X', data)

    class_index = defaultdict(list)
    for i, label in enumerate(labels):
        class_index[label].append(i)

    np.save(f'{zl_path}/{animals_fruits}/class_a', class_index)
