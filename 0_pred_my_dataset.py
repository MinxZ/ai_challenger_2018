import cv2
import numpy as np
from keras.applications import *
from keras.applications.inception_v3 import preprocess_input
from keras.layers import *
from keras.models import *

width = 299
zl_path = '/data/zl'
animals_fruits = 'fruits'

model = load_model(f'{zl_path}/fruits_test/Xception_best.h5')
train_test = 'test'
X = np.load(f'{zl_path}/{animals_fruits}/x_{train_test}.npy')
X = X / 127.5 - 1
features = model.predict(X, batch_size=14, verbose=1)
np.save(f'{zl_path}/{animals_fruits}/label_test', features)

dir_path = f'{zl_path}/ai_challenger_zsl2018_train_test_a_20180321/zsl_a_{animals_fruits}_test_20180321'
images_test = os.listdir(dir_path)

fsplit = open(
    f'{zl_path}/{animals_fruits}/fruits_test_label.txt', 'r', encoding='UTF-8')
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
imagenet_zsl = {}
for label, name in sorted(label2name.items(), key=lambda x: x[0]):
    idx += 1
    imagenet_zsl[idx] = label


def test(path, idx):
    cla = features[idx].argmax()
    zsl_label = imagenet_zsl[cla]
    return zsl_label


fpred = open(f'{zl_path}/{animals_fruits}/ans_fruits_true.txt', 'w')
for idx, path in enumerate(images_test):
    zsl_label = test(path, idx)
    fpred.write(f'{path} {zsl_label}\n')
fpred.close()
