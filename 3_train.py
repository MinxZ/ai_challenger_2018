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
from keras.utils.generic_utils import CustomObjectScope
from tqdm import tqdm


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
        filepath='fc_' + model_name + '.h5', verbose=0, save_best_only=True)
    model_fc.fit(
        features,
        y_train,
        batch_size=128,
        epochs=10000,
        validation_split=0.1,
        callbacks=[checkpointer, early_stopping])


print('\n Loading Datasets. \n')
class_index = np.load('class_a.npy').item()
X = np.load('data.npy')

n = len(X)
n_class = len(class_index)
width = X.shape[1]
y = np.zeros((n, n_class), dtype=np.uint8)

key = -1
for class_name, indexes in sorted(class_index.items(), key=lambda x: x[0]):
    key += 1
    for i in indexes:
        y[i][key] = 1

X, y = unison_shuffled_copies(X, y)

dvi = int(X.shape[0] * 0.9)
x_train = X[:dvi, :, :, :]
y_train = y[:dvi, :]
x_val = X[dvi:, :, :, :]
y_val = y[dvi:, :]

# Fine-tune the model
print(' Build model. \n')

batch_sizes = {"Xception": 32, "InceptionResNetV2": 16, "NASNet": 16}
angles = {"Xception": 20, "InceptionResNetV2": 20, "NASNet": 30}
list_model = {
    "Xception": Xception,
    "InceptionResNetV2": InceptionResNetV2,
    "NASNet": NASNetLarge
}
# model_names = ["Xception"]
# for model_name in model_names:
model_name = "Xception"
MODEL = list_model[model_name]
batch_size = batch_sizes[model_name]

lr = 5e-4  # 1-5e4
epoch = 1e4
reduce_lr_patience = 3  # 1-3
patience = 7  # reduce_lr_patience+1* + 1
angle = angles[model_name]

print(" Fine tune " + model_name + ": \n")

# Build the model
use_imagenet = True
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
    print('Using imagenet weights. \n')
    try:
        model.load_weights('fc_' + model_name + '.h5', by_name=True)
        print('  Load fc_' + model_name + '.h5 successfully.\n')
    except:
        print(' Train fc layer firstly.\n')
        fc_model()
        model.load_weights('fc_' + model_name + '.h5', by_name=True)
        print(' Load fc_' + model_name + '.h5 successfully.\n')

    print("  Optimizer=" + optimizer + " lr=" + str(lr) + " \n")
    model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(lr=lr, momentum=0.9, nesterov=True),
        metrics=['accuracy'])
else:
    print('Not using imagenet weight. \n')
    print("  Optimizer=" + optimizer + " \n")
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy'])

# datagen and val_datagen
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    # preprocessing_function=get_random_eraser(
    #     p=0.2, v_l=0, v_h=255, pixel_level=True),  # 0.1-0.4
    rotation_range=20,  # 10-30
    width_shift_range=0.2,  # 0.1-0.3
    height_shift_range=0.2,  # 0.1-0.3
    shear_range=0.2,  # 0.1-0.3
    zoom_range=0.2,  # 0.1-0.3
    horizontal_flip=True,
    fill_mode='nearest')
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
# val_datagen = ImageDataGenerator()

# callbacks
early_stopping = EarlyStopping(
    monitor='val_loss', patience=patience, verbose=2, mode='auto')
checkpointer = ModelCheckpoint(
    filepath=model_name + '.h5', verbose=0, save_best_only=True)
reduce_lr = ReduceLROnPlateau(
    factor=0.3, patience=reduce_lr_patience, verbose=2)

# Start fitting model
print(' Start fitting. \n')
model.fit_generator(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(x_train) / batch_size,
    validation_data=val_datagen.flow(x_val, y_val, batch_size=batch_size),
    validation_steps=len(x_val) / batch_size,
    epochs=epoch,
    callbacks=[early_stopping, checkpointer, reduce_lr])
