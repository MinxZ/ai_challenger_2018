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


zl_path = '/data/zl/'
path = f'{zl_path}ai_challenger_zsl2018_train_test_a_20180321'
superclasses = ['Animals', 'Fruits']
for dim in [2048, 256, 40]:
    # Write prediction
    fpred = open(f'pred_{dim}.txt', 'w')
    for superclass in superclasses:
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
        for each in lines_label:
            tokens = each.split(', ')
            list_train.append(tokens[0])
            names_train.append(tokens[1])
        list_test = list()
        for i in range(classNum):
            label = 'Label_' + superclass[0] + '_' + str(i + 1).zfill(2)
            if label not in list_train:
                list_test.append(label)

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

        # Load image features

        features_train = np.load(
            f'{zl_path}/{animals_fruits}/features_train_{dim}.npy')
        features_test = np.load(
            f'{zl_path}/{animals_fruits}/features_test_{dim}.npy')
        class_index = np.load(
            f'{zl_path}/{animals_fruits}/class_a_{dim}.npy').item()

        # Calculate prototypes (cluster centers)
        features_test = features_test / np.max(abs(features_train))
        features_train = features_train / np.max(abs(features_train))
        dim_f = features_train.shape[1]
        prototypes_train = np.ndarray((int(classNum / 5 * 4), dim_f))

        dim_a = attrnum[superclass[0]]
        attributes_train = np.ndarray((int(classNum / 5 * 4), dim_a))
        attributes_test = np.ndarray((int(classNum / 5 * 1), dim_a))

        for i in range(len(list_train)):
            label = list_train[i]
            idx = class_index[label]
            prototypes_train[i, :] = np.mean(features_train[idx, :], axis=0)
            attributes_train[i, :] = np.asarray(attributes[label])

        for i in range(len(list_test)):
            label = list_test[i]
            attributes_test[i, :] = np.asarray(attributes[label])

        # Structure learning
        LASSO = models.Lasso(alpha=0.01)
        LASSO.fit(attributes_train.transpose(), attributes_test.transpose())
        W = LASSO.coef_

        # Image prototype synthesis
        prototypes_test = (
            np.dot(prototypes_train.transpose(), W.transpose())).transpose()

        # Prediction

        dir_path = f'{zl_path}/ai_challenger_zsl2018_train_test_a_20180321/zsl_a_{animals_fruits}_test_20180321'
        images_test = os.listdir(dir_path)
        prediction = list()
        for i in range(len(images_test)):
            temp = np.repeat(np.reshape(
                (features_test[i, :]), (1, dim_f)), len(list_test), axis=0)
            distance = np.sum((temp - prototypes_test)**2, axis=1)
            pos = np.argmin(distance)
            prediction.append(list_test[pos])

        for i in range(len(images_test)):
            fpred.write(str(images_test[i]) + ' ' + prediction[i] + '\n')
    fpred.close()
