import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def FC_relu(input_dim, output_dim, name='FC_relu'):
    input_tensor = Input((input_dim,))
    x = Dense(output_dim)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return Model(input_tensor, x, name=name)

def attention_model(timesteps=101, dim=512, unit=128, n_class=5):
    fc_relu = FC_relu(dim, int(dim / 2))

    inputs = Input((timesteps, dim))
    # the weights f of timesteps (timesteps * n_class)
    f_weights = Input((timesteps, n_class))

    x = Lambda(lambda x: tf.unstack(x, axis=1))(inputs)  # len(x)=timestaps
    xs = []
    # dim->dim/2
    for x_i in x:
        xs.append(fc_relu(x_i))
    x = Lambda(lambda x: tf.stack(x, axis=1))(xs)  # len(x)=timestaps
    x = LSTM(unit, return_sequences=True)(x)
    x = Reshape((timesteps, unit))(x)

    # multiply weights f_weights to the result from LSTM
    x = Lambda(lambda x:tf.matmul(x[0], x[1], transpose_a=True))([f_weights, x])

    # final multi-softmax
    x = Lambda(lambda x: tf.unstack(x, axis=1))(x)  # len(x)=timestaps
    xs = []
    # each class with different weights w^n
    for x_i in x:
        xs.append(Dense(1)(x_i))
    x = Lambda(lambda x: tf.stack(x, axis=1))(xs)
    x = Reshape((n_class,))(x)
    x = Activation('softmax')(x)

    return Model(inputs=[inputs,f_weights], outputs=x)