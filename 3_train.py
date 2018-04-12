from __future__ import absolute_import, division, print_function

import multiprocessing as mp
import random

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.applications import *
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import *
from keras.utils import np_utils
from keras.utils.generic_utils import CustomObjectScope
from tqdm import tqdm

from random_eraser import get_random_eraser


def get_features(MODEL, data, batch_size):
    cnn_model = MODEL(input_shape=(width, width, 3),
                      include_top=False,  weights='imagenet', pooling='avg')
    inputs = Input((width, width, 3))
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    cnn_model = Model(inputs, x)
    # cnn_model.load_weights()
    features = cnn_model.predict(data, batch_size=batch_size, verbose=1)
    return features


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def fc_model():
    print(' Start computing ' + model_name + ' bottleneck feature: ')
    features = get_features(MODEL, x_train, batch_size)

    # Training fc models
    inputs = Input(features.shape[1:])
    x = inputs
    x = Dropout(0.5)(x)
    x = Dense(256, activation='elu', name='fc')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_class, activation='softmax', name='predictions')(x)
    model_fc = Model(inputs, x)

    model_fc.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, verbose=1, mode='auto')
    checkpointer = ModelCheckpoint(
        filepath=f'{zl_path}/{animals_fruits}/fc_{model_name}.h5', verbose=0, save_best_only=True)
    model_fc.fit(
        features,
        y_train,
        batch_size=128,
        epochs=10000,
        validation_split=0.1,
        callbacks=[checkpointer, early_stopping])


# Fine-tune the model
batch_sizes = {"MobileNet": 24, "NASNetMobile": 32, "Xception": 14,
               "InceptionResNetV2": 10, "NASNetLarge": 6}
list_model = {
    "MobileNet": MobileNet,
    "Xception": Xception,
    "InceptionResNetV2": InceptionResNetV2,
    "NASNetLarge": NASNetLarge,
    "NASNetMobile": NASNetMobile
}
<<<<<<< HEAD
=======

model_name = "Xception"
MODEL = list_model[model_name]
batch_size = batch_sizes[model_name]

# use_imagenet = True
fine_tune = False
use_imagenet = False
# fine_tune = True
>>>>>>> dcbc0cbe647a4dce94b5eb2601e215466d7f0a79

model_name = "Xception"
MODEL = list_model[model_name]
batch_size = batch_sizes[model_name]

<<<<<<< HEAD
use_imagenet = True
fine_tune = False
# use_imagenet = False
# fine_tune = True


if use_imagenet or fine_tune is True:
    optimizer = 'SGD'
    lr = 1e-4
    lr = lr * batch_size / 32
    opt = SGD(lr=lr, momentum=0.9, decay=1e-6, nesterov=True)

reduce_lr_patience = 5
patience = 10  # reduce_lr_patience+1* + 1
print(
    f'\n Reduce_lr_patience: {reduce_lr_patience} \n\n Patience: {patience} \n ')

=======
if use_imagenet or fine_tune is True:
    optimizer = 'SGD'
    lr = 1e-4
    lr = lr * batch_size / 32
    opt = SGD(lr=lr, momentum=0.9, decay=1e-6, nesterov=True)

reduce_lr_patience = 5
patience = 10  # reduce_lr_patience+1* + 1
print(
    f'\n Reduce_lr_patience: {reduce_lr_patience} \n\n Patience: {patience} \n ')

>>>>>>> dcbc0cbe647a4dce94b5eb2601e215466d7f0a79
zl_path = '/data/zl'
# animals_fruits = 'animals'
animals_fruits = 'fruits'
# animals_fruits = 'fruits_test'

print(f' Training on {animals_fruits} dataset.')
print('\n Loading Datasets. \n')
try:
    y_val = np.load(f'{zl_path}/{animals_fruits}/y_val.npy')
    x_val = np.load(f'{zl_path}/{animals_fruits}/x_val.npy')
    y_train = np.load(f'{zl_path}/{animals_fruits}/y_train.npy')
    x_train = np.load(f'{zl_path}/{animals_fruits}/x_train.npy')
except:
    print('Train val split again.')
    X = np.load(f'{zl_path}/{animals_fruits}/X.npy')
    n = X.shape[0]
    width = X.shape[1]
    try:
        class_index = np.load(f'{zl_path}/{animals_fruits}/class_a.npy').item()
        n_class = len(class_index)
        y = np.zeros((n, n_class), dtype=np.uint8)
        key = -1
        for class_name, indexes in sorted(class_index.items(), key=lambda x: x[0]):
            key += 1
            for i in indexes:
                y[i][key] = 1
    except:
        y = np.load(f'{zl_path}/{animals_fruits}/y.npy')

    X, y = unison_shuffled_copies(X, y)
    dvi = int(X.shape[0] * 0.9)
    x_train = X[:dvi, :, :, :]
    y_train = y[:dvi, :]
    x_val = X[dvi:, :, :, :]
    y_val = y[dvi:, :]

    np.save(f'{zl_path}/{animals_fruits}/y_val', y_val)
    np.save(f'{zl_path}/{animals_fruits}/x_val', x_val)
    np.save(f'{zl_path}/{animals_fruits}/y_train', y_train)
    np.save(f'{zl_path}/{animals_fruits}/x_train', x_train)

n_class = y_val.shape[1]
width = x_val.shape[1]

model_path = f'{zl_path}/{animals_fruits}/{model_name}.h5'


# Build the model
print(f' Build {model_name}. \n')
if use_imagenet == True:
    cnn_model = MODEL(
        include_top=False, input_shape=(width, width, 3), weights='imagenet', pooling='avg')
else:
    cnn_model = MODEL(
        include_top=False, input_shape=(width, width, 3), weights=None, pooling='avg')

inputs = Input((width, width, 3))
x = inputs
# x = Lambda(preprocess_input, name='preprocessing')(x)
x = cnn_model(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='elu', name='fc')(x)
x = Dropout(0.5)(x)
x = Dense(n_class, activation='softmax', name='predictions')(x)
model = Model(inputs=inputs, outputs=x)

# Load weights
if use_imagenet == True:
    print(f' Fine tune {model_name}: \n')
    print(' Using imagenet weights. \n')
    try:
        model.load_weights(
            f'{zl_path}/{animals_fruits}/fc_{model_name}.h5', by_name=True)
        print(f' Load fc_{model_name}.h5 successfully.\n')
    except:
        print(' Train fc layer firstly.\n')
        fc_model()
        model.load_weights(
            f'{zl_path}/{animals_fruits}/fc_' + model_name + '.h5', by_name=True)
        print(f' Load fc_{model_name}.h5 successfully.\n')
        print(f' Fine tune {model_name} with batch size: {batch_size}: \n')

elif fine_tune == True:
    model.load_weights(
        f'{zl_path}/{animals_fruits}/{model_name}_best.h5', by_name=True)
    lr = lr * 0.2
    model_path = f'{model_name}_random_eraser.h5'
    print(f' Fine tune {model_name} with batch size: {batch_size}: \n')

else:
    print(f"\n Train {model_name} with batch size: {batch_size}\n")
    optimizer = 'Adam'
    lr = 0.001
    opt = 'adam'

print(" Optimizer: " + optimizer + " lr: " + str(lr) + " \n")
model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'])


# datagen and val_datagen
datagen = ImageDataGenerator(
<<<<<<< HEAD
    # preprocessing_function=preprocess_input,
    # preprocessing_function=get_random_eraser(
    #     p=0.2, v_l=0, v_h=1, pixel_level=True),  # 0.1-0.4
    rescale=1. / 255,
=======
    preprocessing_function=preprocess_input,
    # preprocessing_function=get_random_eraser(
    #     p=0.2, v_l=0, v_h=1, pixel_level=True),  # 0.1-0.4
    # rescale=1. / 255,
>>>>>>> dcbc0cbe647a4dce94b5eb2601e215466d7f0a79
    rotation_range=40,  # 10-30
    width_shift_range=0.2,  # 0.1-0.3
    height_shift_range=0.2,  # 0.1-0.3
    shear_range=0.2,  # 0.1-0.3
    zoom_range=0.2,  # 0.1-0.3
    horizontal_flip=True,
    fill_mode='nearest')
<<<<<<< HEAD
# val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(rescale=1. / 255)
=======
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
# val_datagen = ImageDataGenerator(rescale=1. / 255)
>>>>>>> dcbc0cbe647a4dce94b5eb2601e215466d7f0a79

# callbacks
early_stopping = EarlyStopping(
    monitor='val_loss', patience=patience, verbose=2, mode='auto')
checkpointer = ModelCheckpoint(
    filepath=model_path, verbose=0, save_best_only=True)
reduce_lr = ReduceLROnPlateau(
    factor=np.sqrt(0.1), patience=reduce_lr_patience, verbose=2)

# Start fitting model.
print(' Start fitting. \n')
model.fit_generator(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(x_train) / batch_size,
    validation_data=val_datagen.flow(x_val, y_val, batch_size=batch_size),
    validation_steps=len(x_val) / batch_size,
    epochs=1e4,
    callbacks=[early_stopping, checkpointer, reduce_lr])
