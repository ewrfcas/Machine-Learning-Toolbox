import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils.vis_utils import plot_model

def attention_model(timesteps=101, dim=512, unit=128, n_class=5):
    inputs = Input((timesteps, dim),name='input_data')
    # the weights f of timesteps (timesteps * n_class)
    f_weights = Input((timesteps, n_class),name='f_weights')

    x = BatchNormalization()(inputs)
    x = LSTM(unit, return_sequences=True)(x)
    x = Reshape((timesteps, unit))(x)

    # multiply weights f_weights to the result from LSTM
    x = Lambda(lambda x: tf.matmul(x[0], x[1], transpose_a=True),name='fweights_mul')([f_weights, x])

    # final multi-softmax
    x = Lambda(lambda x: tf.unstack(x, axis=1),name='unstack1')(x)  # len(x)=timestaps
    xs = []
    # each class with different weights w^n
    for x_i in x:
        xs.append(Dense(1)(x_i))
    x = Lambda(lambda x: tf.stack(x, axis=1),name='stack1')(xs)
    x = Reshape((n_class,))(x)
    x = Activation('softmax')(x)

    return Model(inputs=[inputs, f_weights], outputs=x)

model=attention_model()
plot_model(model,to_file='attention_model.png',show_shapes=False)