import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils.vis_utils import plot_model

def weights_model(timesteps=101, dim=512, unit=128, emotion_embedding_dim=64, n_class=5):
    inputs = Input((timesteps, dim))
    x = BatchNormalization()(inputs)
    x = LSTM(unit, return_sequences=True)(x)
    x = Reshape((timesteps, unit))(x)

    # FC (W^h) for mapping the dim from unit to emotion_embedding_dim
    x = Lambda(lambda x: tf.unstack(x, axis=1),name='unstack1')(x)  # len(x)=timestaps
    xs = []
    # unit->emotion_embedding_dim
    dense2=Dense(emotion_embedding_dim,name='Wh')
    for x_i in x:
        x_temp=dense2(x_i)
        xs.append(x_temp)
    x = xs
    # FC (e={e_1,e_2,...}) for embedding mapping from emotion_embedding_dim to n_class
    xs = []
    # emotion_embedding_dim->n_class
    emotion_vectors=Dense(n_class,activation='softmax', name='emotion_vectors')
    for x_i in x:
        x_temp=emotion_vectors(x_i)
        xs.append(x_temp)
    f_weights = Lambda(lambda x: tf.stack(x, axis=1), name='f_weights')(xs)  # len(x)=timestaps
    x = Lambda(lambda x: tf.reduce_mean(x, axis=1),name='reduce_mean1')(f_weights)

    return Model(inputs=inputs, outputs=x)

model=weights_model()
plot_model(model,to_file='f_weights.png',show_shapes=False)