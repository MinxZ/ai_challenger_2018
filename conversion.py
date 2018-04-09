from __future__ import absolute_import, division, print_function

import multiprocessing as mp
import random
from collections import defaultdict

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils import dataset_util

# %pylab inline


def create_tf_example(encoded_image_data, filename, x_min, x_max, y_min, y_max, classes_text, classes):
    """Creates a tf.Example proto from sample cat image.

    Args:
    encoded_cat_image_data: The jpg encoded data of the cat image.

    Returns:
    example: The created tf.Example.
    """
    image_format = b'jpg'

    xmins = [x_min / width]
    xmaxs = [x_max / width]
    ymins = [y_min / height]
    ymaxs = [y_max / height]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode()),
        'image/source_id': dataset_util.bytes_feature(filename.encode()),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data.tobytes()),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text


dic = {'[': '', ']': '', '\n': ''}


zl_path = '/data/zl'
# zl_path = '/Users/z/zl'
path = f'{zl_path}/ai_challenger_zsl2018_train_test_a_20180321'
# animals_fruits = 'animals'
animals_fruits = 'fruits'
superclasses = ['Animals', 'Fruits']

for superclass in superclasses:
    animals_fruits = str(superclass).lower()
    writer_train = tf.python_io.TFRecordWriter(
        f'{zl_path}/{animals_fruits}/train.record')
    writer_val = tf.python_io.TFRecordWriter(
        f'{zl_path}/{animals_fruits}/val.record')
    dir_path = f'{zl_path}/ai_challenger_zsl2018_train_test_a_20180321/zsl_a_{animals_fruits}_train_20180321'
    fname = f'{dir_path}/zsl_a_{animals_fruits}_train_annotations_labels_20180321.txt'
    data_path = f'{dir_path}/zsl_a_{animals_fruits}_train_images_20180321'
    with open(fname) as f:
        content = f.readlines()
    content = [replace_all(x, dic) for x in content]
    content = [x.split(', ') for x in content]
    data = []
    labels = []

    fpred = open(f'{zl_path}/{animals_fruits}/pred_{superclass}.txt', 'w')
    # The constants
    if superclass[0] == 'H':
        classNum = 30
    else:
        classNum = 50
    testName = {'A': 'a', 'F': 'a', 'V': 'b', 'E': 'b', 'H': 'b'}
    date = '20180321'

    # Load seen/unseen split
    label_list_path = f'{path}/zsl_' + testName[superclass[0]] + '_' + str(superclass).lower() + '_train_' + date\
        + '/zsl_' + testName[superclass[0]] + '_' + str(superclass).lower(
    ) + '_train_annotations_' + 'label_list_' + date + '.txt'
    fsplit = open(label_list_path, 'r', encoding='UTF-8')
    lines_label = fsplit.readlines()
    fsplit.close()
    list_train = list()
    names_train = list()
    for each in lines_label:
        tokens = each.split(', ')
        list_train.append(tokens[0])
        names_train.append(tokens[1])
    label2name = dict(zip(list_train, names_train))

    for x in tqdm(range(len(content))):
        file_name = content[x][6]
        s_img = cv2.imread(data_path + '/' + file_name)
        b, g, r = cv2.split(s_img)       # get b,g,r
        rgb_img = cv2.merge([r, g, b])     # switch it to rgb
        height, width, channel = rgb_img.shape
        x_min = int(content[x][2]) / width
        x_max = int(content[x][4]) / width
        y_min = int(content[x][3]) / height
        y_max = int(content[x][5]) / height
        classes_text_name = label2name[content[x][1]]
        classes_text = [label2name[content[x][1]].encode()]
        classes = [names_train.index(classes_text_name) + 1]
        tf_example = create_tf_example(
            rgb_img, file_name, x_min, x_max, y_min, y_max, classes_text, classes)
        if x % 8 == 0:
            writer_val.write(tf_example.SerializeToString())
        else:
            writer_train.write(tf_example.SerializeToString())
    writer_val.close()
    writer_train.close()
