from keras.layers import *
from keras.models import *

def model(input_shape=(16, 112, 112, 3)):
    inputs = Input(shape=input_shape)
    x = Convolution3D(64, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv1',
                            input_shape=input_shape)(inputs)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           border_mode='valid', name='pool1')(x)
    # 2nd layer group
    x = Convolution3D(128, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv2')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool2')(x)
    # 3rd layer group
    x = Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv3a')(x)
    x = Convolution3D(256, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv3b')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool3')(x)
    # 4th layer group
    x = Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv4a')(x)
    x = Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv4b')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool4')(x)
    # 5th layer group
    x = Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv5a')(x)
    x = Convolution3D(512, 3, 3, 3, activation='relu',
                            border_mode='same', name='conv5b')(x)
    x = ZeroPadding3D(padding=((0, 0), (0, 1), (0, 1)), name='zeropad5')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool5')(x)
    x = Flatten()(x)
    # FC layers group
    x = Dense(4096, activation='relu', name='fc6')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc7')(x)
    x = Dropout(0.5)(x)

    a = Dense(1, activation='sigmoid', name='arousal')(x)
    v = Dense(1, activation='tanh', name='valence')(x)

    model = Model(inputs, [a,v], name='C3D')

    return model

c3d_model=model()
print(c3d_model.summary())

import numpy as np
X=np.random.random((1000,16,112,112,3))
y=np.random.random((1000,2))
y[:,1]=(y[:,1]*2)-1
c3d_model.compile(loss='mse',optimizer='adam')
c3d_model.fit(X,[y[:,0],y[:,1]])