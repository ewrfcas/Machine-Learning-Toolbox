from keras.layers import *
from keras.models import *
from keras import layers
from keras.optimizers import *
import numpy as np

def preprocess_numpy_input(x):
    x /= 127.5
    x -= 1.
    return x

def Residual(x, filters):
    # Skip layer
    shortcut = Conv2D(filters, (1, 1), padding='same')(x)

    # Residual block
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(int(filters / 2), (1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(int(filters / 2), (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same', use_bias=False)(x)
    x = layers.add([x, shortcut])

    return x

def Hourglass(x, level, filters):
    # up layer
    up = Residual(x, filters)

    # low layer
    low = MaxPooling2D()(x)
    low = Residual(low, filters)
    if level>1:
        low = Hourglass(low, level-1, filters)
    else:
        low = Residual(low, filters)
    low = Residual(low, filters)
    low = UpSampling2D()(low)
    x = layers.add([up, low])

    return x

def model(input_shape=(128, 128, 1), labels=40, nstack=6, level=4, filters=256, preprocess=True):
    img_input = Input(shape=input_shape)

    if preprocess:
        x = Lambda(preprocess_numpy_input)(img_input)
    else:
        x = img_input

    # 128*128
    x = Conv2D(64, (7, 7), strides=2, padding='same', use_bias=False)(x)
    # 64*64
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Residual(x, int(filters/2))
    x = MaxPooling2D()(x)
    # 32*32
    x = Residual(x, int(filters/2))
    middle_x = Residual(x, filters)
    outputs=[]

    for i in range(nstack):
        x = Hourglass(middle_x, level, filters)
        x = Residual(x, filters)
        x = Conv2D(filters, (1, 1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        temp_output = Conv2D(labels, (1, 1), padding='same', name='nstack_'+str(i+1))(x)
        outputs.append(temp_output)

        if i < nstack-1:
            x = Conv2D(filters, (1, 1), padding='same')(x)
            temp_output_ = Conv2D(filters, (1, 1), padding='same')(temp_output)
            middle_x = layers.add([middle_x, x, temp_output_])

    # Create model.
    model = Model(img_input, outputs, name='hourglass')

    return model

# test test
model=model((64,64,3),labels=10,preprocess=False)
optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mse', optimizer=optimizer)
X_train=np.random.random((1000,64,64,3))
y=np.random.random((1000,16,16,10))
model.fit(X_train,[y,y,y,y,y,y],verbose=1)