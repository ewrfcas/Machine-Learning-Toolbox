import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import applications

def model(timesteps=600000, output_type='pred'):
    inputs = Input((timesteps,))
    x = Reshape((timesteps, 1))(inputs)
    x = Conv1D(filters=16, kernel_size=64, strides=2, name='conv1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=8)(x)

    x = Conv1D(filters=32, kernel_size=32, strides=2, name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=8)(x)

    x = Conv1D(filters=64, kernel_size=16, strides=2, name='conv3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = Activation('relu')(x)
    x = Conv1D(filters=128, kernel_size=8, strides=2, name='conv4')(x)
    x = BatchNormalization(name='bn4')(x)
    x = Activation('relu')(x)
    x = Conv1D(filters=256, kernel_size=4, strides=2, name='conv5')(x)
    x = BatchNormalization(name='bn5')(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=4)(x)

    if output_type=='feat':
        output = GlobalAveragePooling1D()(x)
    else:
        x = Conv1D(filters=512, kernel_size=4, strides=2, name='conv6')(x)
        x = BatchNormalization(name='bn6')(x)
        x = Activation('relu')(x)

        x = Conv1D(filters=1024, kernel_size=4, strides=2, name='conv7')(x)
        x = BatchNormalization(name='bn7')(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)

        x1 = Conv1D(1, kernel_size=16, activation='sigmoid', name='last_conv_1')(x)
        x1 = Reshape((1,))(x1)
        x2 = Conv1D(1, kernel_size=16, activation='tanh', name='last_conv_2')(x)
        x2 = Reshape((1,))(x2)
        output = [x1,x2]

    return Model(inputs=inputs, outputs=output)

model=model(output_type='feat')
model.summary()