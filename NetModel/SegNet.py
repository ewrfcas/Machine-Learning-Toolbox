# used for DeMesh
from keras.models import *
from keras.layers import *

def SegNet(image_size=(220, 178, 3)):
    inputs = Input(image_size)
    mask = Input(image_size)

    # encoder
    x = Conv2D(64, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(512, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    # decoder
    x = UpSampling2D()(x)
    x = Conv2D(512, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D()(x)
    x = ZeroPadding2D((1, 0))(x)
    x = Conv2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D()(x)
    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D()(x)
    x = ZeroPadding2D((2,1))(x)
    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x1 = Conv2D(3, 1, activation='sigmoid')(x)
    x2 = Multiply()([mask, x1])

    return Model(inputs=[inputs, mask], outputs=[x1, x2])

model=SegNet()
model.summary()