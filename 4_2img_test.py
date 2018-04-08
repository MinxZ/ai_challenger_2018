import os

import cv2
import numpy as np

animals_fruits = 'animals'
train_test = 'test'
zl_path = '/data/zl'
dir_path = f'{zl_path}/ai_challenger_zsl2018_train_test_a_20180321/zsl_a_{animals_fruits}_test_20180321'
images_test = os.listdir(dir_path)

X = np.load(f'{zl_path}/{animals_fruits}/x_{train_test}_resnet101.npy')
for idx, path in enumerate(images_test):
    s_img = X[idx]
    b, g, r = cv2.split(s_img)       # get b,g,r
    rgb_img = cv2.merge([r, g, b])     # switch it to rgb
    cv2.imwrite(f'{zl_path}/img_test/{path}', rgb_img)
