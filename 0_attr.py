#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 challenger.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Baseline codes for zero-shot learning task.
This python script is the baseline to implement zero-shot learning on each super-class.
The command is:     python MDP.py Animals
The only parameter is the super-class name.
This method is from the paper
@inproceedings{zhao2017zero,
  title={Zero-shot learning posed as a missing data problem},
  author={Zhao, Bo and Wu, Botong and Wu, Tianfu and Wang, Yizhou},
  booktitle={Proceedings of ICCV Workshop},
  pages={2616--2622},
  year={2017}
}
Cite the paper, if you use this code.
"""

from __future__ import absolute_import, division, print_function

import os

import numpy as np
import sklearn.linear_model as models


def attrstr2list(s):
    # Convert strings to attributes
    s = s[1:-2]
    tokens = s.split()
    attrlist = list()
    for each in tokens:
        attrlist.append(float(each))
    return attrlist


zl_path = '/data/zl'
zl_path = '/Users/z/zl'
path = f'{zl_path}/ai_challenger_zsl2018_train_test_a_20180321'

superclass = 'Animals'
superclass = 'Fruits'
animals_fruits = str(superclass).lower()

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
names_train_ch = list()
for each in lines_label:
    tokens = each.split(', ')
    list_train.append(tokens[0])
    names_train.append(tokens[1])
    names_train_ch.append(tokens[2])
list_test = list()
for i in range(classNum):
    label = 'Label_' + superclass[0] + '_' + str(i + 1).zfill(2)
    if label not in list_train:
        list_test.append(label)


list_name = dict(zip(list_train, names_train))
list_name_ch = dict(zip(list_train, names_train_ch))

# Load attributes
attrnum = {'A': 123, 'F': 58, 'V': 81, 'E': 75, 'H': 22}

attributes_per_class_path = f'{path}/zsl_' + testName[superclass[0]] + '_' + str(superclass).lower() + '_train_' + date \
    + '/zsl_' + testName[superclass[0]] + '_' + str(superclass).lower() \
    + '_train_annotations_' + 'attributes_per_class_' + date + '.txt'

fattr = open(attributes_per_class_path, 'r', encoding='UTF-8')
lines_attr = fattr.readlines()
fattr.close()
attributes = dict()
for each in lines_attr:
    tokens = each.split(', ')
    label = tokens[0]
    attr = attrstr2list(tokens[1])
    if not (len(attr) == attrnum[superclass[0]]):
        print('attributes number error\n')
        exit()
    attributes[label] = attr


attributes_names_path = f'{path}/zsl_' + testName[superclass[0]] + '_' + str(superclass).lower() + '_train_' + date \
    + '/zsl_' + testName[superclass[0]] + '_' + str(superclass).lower() \
    + '_train_annotations_' + 'attribute_list_' + date + '.txt'

fattr = open(attributes_names_path, 'r', encoding='UTF-8')
lines_attr_name = fattr.readlines()
fattr.close()
attributes_names = dict()
attributes_names_ch = dict()
for each in lines_attr_name:
    tokens = each.split(', ')
    label = tokens[0]
    attributes_names[int(label[-3:])] = tokens[1]
    attributes_names_ch[int(label[-3:])] = tokens[2]

# rm attr_animals.txt

fpred = open(f'{zl_path}/attr_{animals_fruits}_ch.txt', 'w')
for label, attrs in sorted(attributes.items(), key=lambda x: x[0]):
    if label in list_train:
        name = list_name[label] + ' ' + list_name_ch[label]
    else:
        name = 'test'
    fpred.write(f'{label} {name}\n')
    for idx, attr in enumerate(attrs):
        if attr > 0:
            fpred.write(f'{attr} {attributes_names_ch[idx+1]}')
    fpred.write('------------------------------------ \n\n')
fpred.close()

fpred = open(f'{zl_path}/attr_{animals_fruits}_en.txt', 'w')
for label, attrs in sorted(attributes.items(), key=lambda x: x[0]):
    if label in list_train:
        name = list_name[label] + ' ' + list_name_ch[label]
    else:
        name = 'test'
    fpred.write(f'{label} {name}\n\n')
    for idx, attr in enumerate(attrs):
        if attr > 0:
            fpred.write(f'{attr} {attributes_names[idx+1]}')
    fpred.write('------------------------------------ \n\n')
fpred.close()
