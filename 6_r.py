import cv2
import numpy as np
from keras.applications import *
from keras.applications.inception_v3 import preprocess_input
from keras.layers import *
from keras.models import *

width = 299
MODEL = InceptionResNetV2
batch_size = 16

try:
    animals_fruits = 'animals'
    train_test = 'test'
    zl_path = '/data/zl'
    dir_path = f'{zl_path}/ai_challenger_zsl2018_train_test_a_20180321/zsl_a_{animals_fruits}_test_20180321'
    images_test = os.listdir(dir_path)
    X = np.load(f'{zl_path}/{animals_fruits}/imagenet_label_test.npy')
except:
    # Build the model
    print(f'    Build InceptionResNetV2.')
    cnn_model = MODEL(include_top=True, input_shape=(
        width, width, 3), weights='imagenet', pooling='avg')
    inputs = Input((width, width, 3))
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    model = Model(inputs=inputs, outputs=x)

    # for animals_fruits in ['animals', 'fruits']:
    for animals_fruits in ['animals']:
        print(f'\n {animals_fruits}: ')
        train_test = 'test'
        print(f'  {train_test}: ')
        X = np.load(f'{zl_path}/{animals_fruits}/x_{train_test}.npy')
        features = model.predict(X, batch_size=batch_size, verbose=1)
        np.save(f'{zl_path}/{animals_fruits}/imagenet_label_test', features)


def t(path):
    sample_code = images_test.index(f'{path}.jpg')
    sample_feature = features[sample_code]
    top_5 = sample_feature.copy().argsort()[::-1][:5]
    print(top_5)
    sample_feature2 = sample_feature.copy()
    sample_feature2.sort()
    print(sample_feature2[::-1][:5])


zsl_imagenet = {'Label_A_02': [0, 1, 2, 3, 4],
                'Label_A_05': [5, 6],
                'Label_A_08': [7],
                'Label_A_14': [8],
                'Label_A_20': [9, 10, 11, 12],
                'Label_A_29': [13],
                'Label_A_31': [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
                'Label_A_35': [27, 28, 29, 30],
                'Label_A_39': [31],
                'Label_A_41': [32, 33, 34]
                }
imagenet_zsl = {}
for labels in zsl_imagenet.keys():
    for x in zsl_imagenet[labels]:
        imagenet_zsl[x] = labels


def test(path):
    sample_code = images_test.index(path)
    sample_feature = features[sample_code][candidate_labels].argmax()
    zsl_label = imagenet_zsl[sample_feature]
    return zsl_label


fpred = open(f'animal_test.txt', 'w')
for idx, path in enumerate(images_test):
    zsl_label = test(path)
    fpred.write(f'{path} {zsl_label}\n')
fpred.close()
