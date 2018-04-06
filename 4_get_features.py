import numpy as np
from keras.applications import *
from keras.applications.inception_v3 import preprocess_input
from keras.layers import *
from keras.models import *

width = 299
model_name = "Xception"
MODEL = Xception

# Build the model
print(f' Build {model_name}. \n')
cnn_model = MODEL(
    include_top=False, input_shape=(width, width, 3), weights=weights, pooling='avg')
inputs = Input((width, width, 3))
x = inputs
x = cnn_model(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='elu', name='fc')(x)
x = Dropout(0.5)(x)
x = Dense(n_class, activation='softmax', name='predictions')(x)
model = Model(inputs=inputs, outputs=x)

batch_size = 10
for train_test in ['train', 'test']:
    if train_test == 'test':
        X = np.load('x_test.npy')
    else:
        X = np.load('data.npy')
    features = model.predict(X, batch_size=batch_size, verbose=1)
    np.save(f'features_{model_name}_{train_test}', features)
