from keras.layers import *
from keras.models import *

#hid_node is a list
#if the last_activation_func is sigmoid,x must be minmax normalized
def autoencoder(x,hid_node,activation_func='relu',last_activation_func='sigmoid',batch_size=64,epochs=150):
    input_dim=x.shape[1]
    input_data = Input(shape=(input_dim,))
    # Encoder
    encoder_out = Dense(hid_node[0], activation=activation_func)(input_data)
    for i in range(1,len(hid_node)):
        encoder_out = Dense(hid_node[i], activation=activation_func)(encoder_out)
    #Decoder
    for i in range(len(hid_node)-2,-1,-1):
        encoder_out = Dense(hid_node[i], activation=activation_func)(encoder_out)
    decoder_out = Dense(input_dim, activation=last_activation_func)(encoder_out)
    #Model
    encoder = Model(inputs=input_data, outputs=encoder_out)
    encoder_decoder=Model(inputs=input_data, outputs=decoder_out)
    encoder_decoder.compile(loss='mse', optimizer='adam')
    encoder_decoder.fit(x=x, y=x, batch_size=batch_size, epochs=epochs,shuffle=True)

    return encoder,encoder_decoder


