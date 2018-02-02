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


def FC_linear(input_dim, output_dim, name='FC_linear'):
    input_tensor = Input((input_dim,))
    x = Dense(output_dim)(input_tensor)
    return Model(input_tensor, x, name=name)


def FC_softmax(input_dim, output_dim, name='FC_softmax'):
    input_tensor = Input((input_dim,))
    x = Dense(output_dim, activation='softmax')(input_tensor)
    return Model(input_tensor, x, name=name)


def weights_model(timesteps=101, dim=512, unit=128, emotion_embedding_dim=64, n_class=5):
    fc_relu = FC_relu(dim, int(dim / 2))
    fc_linear = FC_linear(unit, emotion_embedding_dim)
    fc_softmax = FC_softmax(emotion_embedding_dim, n_class)
    inputs = Input((timesteps, dim))

    x = Lambda(lambda x: tf.unstack(x, axis=1))(inputs)  # len(x)=timestaps
    xs = []
    # dim->dim/2
    for x_i in x:
        xs.append(fc_relu(x_i))
    x = Lambda(lambda x: tf.stack(x, axis=1))(xs)  # len(x)=timestaps
    x = LSTM(unit, return_sequences=True)(x)
    x = Reshape((timesteps, unit))(x)

    # FC (W^h) for mapping the dim from unit to emotion_embedding_dim
    x = Lambda(lambda x: tf.unstack(x, axis=1))(x)  # len(x)=timestaps
    xs = []
    # unit->emotion_embedding_dim
    for x_i in x:
        xs.append(fc_linear(x_i))
    x = xs
    # FC (e={e_1,e_2,...}) for embedding mapping from emotion_embedding_dim to n_class
    xs = []
    # emotion_embedding_dim->n_class
    for x_i in x:
        xs.append(fc_softmax(x_i))
    f_weights = Lambda(lambda x: tf.stack(x, axis=1), name='f_weights')(xs)  # len(x)=timestaps
    x = Lambda(lambda x: tf.reduce_mean(x, axis=1))(f_weights)

    return Model(inputs=inputs, outputs=x)