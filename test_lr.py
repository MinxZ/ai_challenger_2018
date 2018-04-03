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


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


print('\n Loading Datasets. \n')
class_index = np.load('class_a.npy').item()
X = np.load('data.npy')
n = X.shape[0]
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

batch_sizes = {"MobileNet": 24, "NASNetMobile": 32, "Xception": 14,
               "InceptionResNetV2": 12, "NASNetLarge": 6}
list_model = {
    "MobileNet": MobileNet,
    "Xception": Xception,
    "InceptionResNetV2": InceptionResNetV2,
    "NASNetLarge": NASNetLarge,
    "NASNetMobile": NASNetMobile
}
use_imagenet = False
weights = None

epoch = 3
reduce_lr_patience = 20  # 1-3
patience = 40  # reduce_lr_patience+1* + 1

#
# optimizer = 'SGD'
# lr = 1e-4
# optimizer = 'RMSprop'
# lr = 1e-4
optimizer = 'Adam'
lr = 2e-4
optimizers = ['Adam', 'SGD']
model_names = ["MobileNet", "Xception"]

for model_name in model_names:
    # model_name = "MobileNet"
    MODEL = list_model[model_name]
    batch_size = batch_sizes[model_name]
    # Build the model
    print(f' Build {model_name}. \n')
    cnn_model = MODEL(
        include_top=False, input_shape=(width, width, 3), weights=weights, pooling='avg')
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
        print(" Fine tune " + model_name + ": \n")
        print('\n Using imagenet weights. \n')
        try:
            model.load_weights('fc_' + model_name + '.h5', by_name=True)
            print('  Load fc_' + model_name + '.h5 successfully.\n')
        except:
            print(' Train fc layer firstly.\n')
            fc_model()
            model.load_weights('fc_' + model_name + '.h5', by_name=True)
            print(' Load fc_' + model_name + '.h5 successfully.\n')
    for optimizer in optimizers:
        if optimizer == "Adam":
            lr = 1e-5
        else:
            lr = 2e-4
        lr = lr * batch_size / 32

        print("\n Optimizer=" + optimizer + " lr=" + str(lr) + " \n")
        if optimizer == "Adam":
            model.compile(
                loss='categorical_crossentropy',
                optimizer=Adam(lr=lr),
                metrics=['accuracy'])
        elif optimizer == "SGD":
            model.compile(
                loss='categorical_crossentropy',
                optimizer=SGD(lr=lr, momentum=0.9, nesterov=True),
                metrics=['accuracy'])
        elif optimizer == 'RMSprop':
            model.compile(
                loss='categorical_crossentropy',
                optimizer=RMSprop(lr=lr),
                metrics=['accuracy'])

        print(f" Train {model_name} with batch size: {batch_size}\n")
        # datagen and val_datagen
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            # preprocessing_function=get_random_eraser(
            #     p=0.2, v_l=0, v_h=255, pixel_level=True),  # 0.1-0.4
            # rescale=1. / 255,
            rotation_range=40,  # 10-30
            width_shift_range=0.2,  # 0.1-0.3
            height_shift_range=0.2,  # 0.1-0.3
            shear_range=0.2,  # 0.1-0.3
            zoom_range=0.2,  # 0.1-0.3
            horizontal_flip=True,
            fill_mode='nearest')
        val_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input)
        # val_datagen = ImageDataGenerator(rescale=1. / 255)

        # callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=patience, verbose=2, mode='auto')
        checkpointer = ModelCheckpoint(
            filepath=f'{model_name}_{optimizer}_{lr}.h5', verbose=0, save_best_only=True)
        reduce_lr = ReduceLROnPlateau(
            factor=0.2, patience=reduce_lr_patience, verbose=2)

        # Start fitting model
        print(' Start fitting. \n')
        model.fit_generator(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(x_train) / batch_size,
            validation_data=val_datagen.flow(
                x_val, y_val, batch_size=batch_size),
            validation_steps=len(x_val) / batch_size,
            epochs=epoch,
            callbacks=[early_stopping, checkpointer, reduce_lr])
