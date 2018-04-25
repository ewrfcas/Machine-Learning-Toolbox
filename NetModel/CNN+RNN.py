import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras_vggface.vggface import VGGFace

def model(vgg_model, input_shape=(224,224,3), CNN_output_dim=512, timesteps=8, unit=256, ac='sigmoid'):
    inputs = Input((timesteps, input_shape[0], input_shape[1], input_shape[2]))
    masking_len = Input((timesteps,CNN_output_dim))
    x = Lambda(lambda x:tf.unstack(x,axis=1),name='unstack')(inputs)
    xs=[]
    for xi in x:
        xs.append(vgg_model(xi))
    x = Lambda(lambda x:tf.stack(x,axis=1),name='stack')(xs)
    x = Lambda(lambda x: x[0] * x[1])([x, masking_len])
    x = Masking(mask_value=0)(x)
    x = LSTM(unit, return_sequences=True,name='LSTM')(x)
    if ac == 'tanh':
        output = Dense(1, activation='tanh',name='output1')(x)
    elif ac == 'tanh+sigmoid':
        x1 = Dense(1, activation='sigmoid',name='output1')(x)
        x2 = Dense(1, activation='tanh',name='output2')(x)
        output = [x1, x2]
    else:
        output = Dense(1, activation='sigmoid',name='output1')(x)

    return Model(inputs=[inputs,masking_len], outputs=output)


VGGFace_model = VGGFace(model='vgg16', weights=None, include_top=False, input_shape=(224,224,3), pooling='avg')
model=model(VGGFace_model)
model.summary()