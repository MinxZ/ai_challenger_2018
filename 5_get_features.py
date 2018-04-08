import numpy as np
from keras.applications import *
from keras.layers import *
from keras.models import *

width = 299
model_name = "Xception_best"
MODEL = Xception
weights = None
batch_size = 12
zl_path = '/data/zl'

# from keras.applications.inception_v3 import preprocess_input


def preprocess_input(x):
    x /= 255.
    return x


animals_fruits = 'animals'
# Build the model
dim = 256
print(f'    Build {model_name}.')
cnn_model = MODEL(
    include_top=False, input_shape=(width, width, 3), weights=None, pooling='avg')
inputs = Input((width, width, 3))
x = inputs
x = Lambda(preprocess_input, name='preprocessing')(x)
x = cnn_model(x)
x = Dropout(0.5)(x)
x = Dense(dim, activation='elu', name='fc')(x)
model = Model(inputs=inputs, outputs=x)
model.load_weights(f'{zl_path}/{animals_fruits}/{model_name}.h5', by_name=True)

animals_fruits = 'animals'
print(f'\n {animals_fruits}: ')
train_test = 'test'
print(f'  {train_test}: ')
X = np.load(f'{zl_path}/{animals_fruits}/x_{train_test}.npy')
features = model.predict(X, batch_size=batch_size, verbose=1)
np.save(f'{zl_path}/{animals_fruits}/features_{train_test}', features)

# for animals_fruits in ['animals', 'fruits']:
#     print(f'\n {animals_fruits}: ')
#     for train_test in ['train', 'test']:
#         print(f'  {train_test}: ')
#         X = np.load(f'{animals_fruits}/x_{train_test}.npy')
#         features = model.predict(X, batch_size=batch_size, verbose=1)
#         np.save(f'{zl_path}/{animals_fruits}/features_{train_test}', features)
