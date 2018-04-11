import numpy as np
from keras.applications import *
from keras.layers import *
from keras.models import *


def preprocess_input(x):
    x /= 255.
    return x


width = 299
model_name = "Xception_best"
MODEL = Xception
batch_size = 12
zl_path = '/data/zl'

# Build the model
print(f'    Build {model_name}.')
cnn_model = MODEL(
    include_top=False, input_shape=(width, width, 3), weights=None, pooling='avg')
inputs = Input((width, width, 3))
x = inputs
x = Lambda(preprocess_input, name='preprocessing')(x)
x = cnn_model(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='elu', name='fc')(x)
model = Model(inputs=inputs, outputs=x)

animals_fruits = 'animals'
for animals_fruits in ['animals', 'fruits']:
    model.load_weights(
        f'{zl_path}/{animals_fruits}/{model_name}.h5', by_name=True)
    print(f'\n {animals_fruits}: ')
    for train_test in ['train', 'test']:
        print(f'  {train_test}: ')
        if train_test == 'test':
            data_path = f'{zl_path}/{animals_fruits}/x_{train_test}.npy'
        else:
            data_path = f'{zl_path}/{animals_fruits}/X.npy'
        X = np.load(data_path)
        features = model.predict(X, batch_size=batch_size, verbose=1)
        np.save(f'{zl_path}/{animals_fruits}/features_{train_test}', features)
