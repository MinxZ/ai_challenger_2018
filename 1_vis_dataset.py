import os

import cv2
import numpy as np

animals_fruits = 'fruits_test'
zl_path = '/data/zl'
dir_path = f'{zl_path}/{animals_fruits}'

X = np.load(f'{zl_path}/{animals_fruits}/x_train.npy')
y = np.load(f'{zl_path}/{animals_fruits}/y_train.npy')
if not os.path.exists(f'{zl_path}/img_test/'):
    os.makedirs(f'{zl_path}/img_test/')
for idx in range(X.shape[0]):
    s_img = X[idx]
    b, g, r = cv2.split(s_img)       # get b,g,r
    rgb_img = cv2.merge([r, g, b])     # switch it to rgb
    label = y[idx].argmax()
    cv2.imwrite(f'{zl_path}/img_test/{label}_{idx}.jpg', rgb_img)
