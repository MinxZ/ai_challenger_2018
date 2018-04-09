
from __future__ import absolute_import, division, print_function

import os

import numpy as np

# zl_path = '/Users/z/zl'
zl_path = '/data/zl'
path = f'{zl_path}/ai_challenger_zsl2018_train_test_a_20180321'
superclasses = ['Animals', 'Fruits']
dim = 256


# Write prediction
for superclass in superclasses:
    animals_fruits = str(superclass).lower()
    fpred = open(f'{zl_path}/{animals_fruits}/data/label_map.pbtxt', 'w')
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

    for i, name in enumerate(names_train):
        fpred.write('item {\n')
        fpred.write(f'    id: {i+1}\n')
        fpred.write(f"    name: '{name}'\n")
        fpred.write('}\n')
        fpred.write('\n\n')
    fpred.close()
