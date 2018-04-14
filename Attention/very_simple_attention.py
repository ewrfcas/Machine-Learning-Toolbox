from keras.layers import *
from keras.optimizers import *

def VerySimpleAttention(inputs,timesteps,reduce_mean=True):
    a = Lambda(lambda x:K.permute_dimensions(x,(0,2,1)))(inputs)
    a = Dense(timesteps, activation='softmax', name='attention_W')(a)
    a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
    a = RepeatVector(inputs.shape[-1])(a)
    a_probs = Lambda(lambda x:K.permute_dimensions(x,(0,2,1)))(a)
    a_weights= Lambda(lambda x:tf.reduce_mean(x,axis=-1), name='attention_weights')(a_probs)
    x = Multiply(name='attention_mul')([inputs, a_probs])
    if reduce_mean:
        x = Lambda(lambda x:tf.reduce_mean(x,axis=1))(x)
    else:
        x = Flatten()(x)
    return x, a_weights