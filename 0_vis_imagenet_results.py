import os
from shutil import copyfile

import cv2
import numpy as np
from tqdm import tqdm

# animals_fruits = 'animals'
animals_fruits = 'fruits'
train_test = 'test'
zl_path = '/Users/z/zl'
dir_path = f'{zl_path}/ai_challenger_zsl2018_train_test_a_20180321/zsl_a_{animals_fruits}_test_20180321'

fname = f'{zl_path}/{animals_fruits}_attr/ans_{animals_fruits}_true.txt'
with open(fname) as f:
    content = f.readlines()
content = [x.strip().split(' ') for x in content]

label2name = {'Label_A_02': '猫',
              'Label_A_05': '狮子',
              'Label_A_08': '马',
              'Label_A_14': '小熊猫',
              'Label_A_20': '驴',
              'Label_A_29': '鹅',
              'Label_A_31': '燕',
              'Label_A_35': '海葵',
              'Label_A_39': '鸭嘴兽',
              'Label_A_41': '鲨鱼'}

if animals_fruits == 'fruits':
    fname = f'{zl_path}/{animals_fruits}_attr/{animals_fruits}_test_label.txt'
    with open(fname) as f:
        content1 = f.readlines()
    content1 = [x.strip().split(', ') for x in content1]

    label2name = {}
    for x, line in enumerate(content1):
        label2name[line[0]] = line[2]

os.mkdir(f'{zl_path}/img_test')
for label in label2name.keys():
    os.mkdir(f'{zl_path}/img_test/{label}')
for idx, path in tqdm(enumerate(content)):
    copyfile(f'{dir_path}/{path[0]}',
             f'{zl_path}/img_test/{path[1]}/{path[0]}')
    # s_img = cv2.imread(f'{dir_path}/{path[0]}')
    # cv2.imwrite(f'{zl_path}/img_test/{label2name[path[1]]}_{path[0]}', s_img)
