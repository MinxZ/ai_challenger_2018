import numpy as np
from keras.applications import *
from keras.layers import *
from keras.models import *

width = 299
model_name = "Xception_best"
MODEL = Xception
weights = None
batch_size = 12

# from keras.applications.inception_v3 import preprocess_input


def preprocess_input(x):
    x /= 255.
    return x

# Build the model


def model_load(dim):
    print(f'    Build {model_name}.')
    cnn_model = MODEL(
        include_top=False, input_shape=(width, width, 3), weights=None, pooling='avg')
    if dim == 2048:
        inputs = Input((width, width, 3))
        x = inputs
        x = Lambda(preprocess_input, name='preprocessing')(x)
        x = cnn_model(x)
        model = Model(inputs=inputs, outputs=x)
        model.load_weights(f'{animals_fruits}/{model_name}.h5', by_name=True)
    elif dim == 256:
        inputs = Input((width, width, 3))
        x = inputs
        x = Lambda(preprocess_input, name='preprocessing')(x)
        x = cnn_model(x)
        x = Dropout(0.5)(x)
        x = Dense(dim, activation='elu', name='fc')(x)
        model = Model(inputs=inputs, outputs=x)
        model.load_weights(f'{animals_fruits}/{model_name}.h5', by_name=True)
    elif dim == 40:
        inputs = Input((width, width, 3))
        x = inputs
        x = Lambda(preprocess_input, name='preprocessing')(x)
        x = cnn_model(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='elu', name='fc')(x)
        x = Dropout(0.5)(x)
        x = Dense(dim, activation='softmax', name='predictions')(x)
        model = Model(inputs=inputs, outputs=x)
        model.load_weights(f'{animals_fruits}/{model_name}.h5', by_name=True)
    return model


for animals_fruits in ['animals', 'fruits']:
    print(f'{animals_fruits}: ')
    for train_test in ['train', 'test']:
        print(f'  {train_test}: ')
        if train_test == 'test':
            X = np.load(f'{animals_fruits}/x_test.npy')
        else:
            X = np.load(f'{animals_fruits}/data.npy')
        for dim in [2048, 256, 40]:
            print(f'\n    dim: {dim}')
            model = model_load(dim)
            features = model.predict(X, batch_size=batch_size, verbose=1)
            np.save(f'{animals_fruits}/features_{train_test}_{dim}', features)
