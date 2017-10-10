from keras.layers import *
from keras.models import *
from keras.optimizers import *
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class DCGAN():
    def __init__(self,image_size=(64,64,1),random_vector_size=100):
        self.image_size=image_size
        self.random_vector_size=random_vector_size

        # discriminator
        self.discriminator = self.Discriminator()
        self.optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        # generator
        self.generator = self.Generator()

        # adversarial
        self.adversarial = Sequential()
        self.adversarial.add(self.generator)
        self.adversarial.add(self.discriminator)
        self.optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.adversarial.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

    def fit(self,path,batch_size=64,epochs=2000,draw=False):
        train_list=np.array(os.listdir(path))
        train_index = np.arange(len(train_list))
        self.path=path
        self.batch_size=batch_size
        self.epochs=epochs
        self.batch_index=None
        for i in range(epochs):
            print(str(i + 1) + '/' + str(epochs))
            images_train=self.train_generator(train_list,train_index)
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, images_fake), axis=0)
            y = np.ones([2 * batch_size, 1])
            y[batch_size:, :] = 0
            d_loss_acc=self.discriminator.train_on_batch(x, y)
            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            g_loss_acc=self.adversarial.train_on_batch(noise, y)
            print('discriminator_loss/acc:' + str(d_loss_acc) + ' generator_loss/acc:' + str(g_loss_acc))
            if (i+1)%10==0:
                if draw:
                    self.plot_images(path='fake',step=i+1)
        return self

    def fit_total(self,X,batch_size=64,epochs=2000,draw=False):
        train_index = np.arange(X.shape[0])
        self.X=X
        self.batch_size=batch_size
        self.epochs=epochs
        self.batch_index=None
        for i in range(epochs):
            print(str(i + 1) + '/' + str(epochs))
            images_train=self.train_generator_data(train_index)
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, images_fake), axis=0)
            y = np.ones([2 * batch_size, 1])
            y[batch_size:, :] = 0
            d_loss_acc=self.discriminator.train_on_batch(x, y)
            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            g_loss_acc=self.adversarial.train_on_batch(noise, y)
            print('discriminator_loss/acc:' + str(d_loss_acc) + ' generator_loss/acc:' + str(g_loss_acc))
            if (i+1)%200==0:
                if draw:
                    self.plot_images(path='fake',step=i+1,image_size=self.image_size)
        return self

    def train_generator(self,train_list,train_index):
        batch_index=np.random.choice(train_index,size=self.batch_size,replace=False)
        train_list_batch=train_list[batch_index]
        images_train=[]
        for s in train_list_batch:
            img = cv.imread(self.path + "/" + s,cv.IMREAD_GRAYSCALE)
            img = cv.resize(img, (self.image_size[0],self.image_size[1]))
            img=self.preprocess_input(img)
            images_train.append(img)
        images_train=np.reshape(images_train, (self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2]))
        return images_train

    def train_generator_data(self,train_index):
        batch_index=np.random.choice(train_index,size=self.batch_size,replace=False)
        train_list_batch=self.X[batch_index,:]
        images_train=[]
        for i in range(train_list_batch.shape[0]):
            img=train_list_batch[i,:]
            img=cv.resize(img,(self.image_size[0],self.image_size[1]))
            images_train.append(img)
        images_train=np.reshape(images_train, (self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2]))
        images_train=self.preprocess_input_to_tanh(images_train)
        return images_train

    def preprocess_input(self,x):
        x = x/255.
        x = x*2.
        x = x-1.
        return x

    def preprocess_output(self,x):
        x = x+1.
        x = x/2.
        x = x*255.
        return x

    def preprocess_input_to_tanh(self,x):
        x = x*2.
        x = x-1.
        return x

    def plot_images(self, samples=16, noise=None, step=0, image_size=(64, 64, 1), path=None,show=False):
        if noise is None:
            noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
        filename = "fake_step" + str(step) + ".png"
        images = self.preprocess_output(self.generator.predict(noise))
        plt.figure(figsize=(6, 6))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i + 1)
            image = images[i, :, :, :]
            image = np.reshape(image, [image_size[0],image_size[1]])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if path != None:
            filename = path + '/' + filename
        if show:
            plt.show()
        plt.savefig(filename)
        plt.close('all')

    def Discriminator(self):
        input = Input(shape=(self.image_size[0], self.image_size[1], self.image_size[2]))
        x = Conv2D(32, (3, 3), strides=2, padding='same')(input)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)

        x = Conv2D(64, (3, 3), strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)

        x = Conv2D(128, (3, 3), strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)

        x = Conv2D(256, (3, 3), strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)

        x = Flatten()(x)
        output = Dense(1, activation='sigmoid')(x)

        return Model(inputs=input, outputs=output, name='Discriminator')

    def Generator(self):
        dim = int(self.image_size[0] / 16)
        origin_channel = self.image_size[2]
        fake_input = Input(shape=(self.random_vector_size,))
        x = Dense(dim * dim * 256)(fake_input)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)
        x = Reshape((dim, dim, 256))(x)
        x = Dropout(0.3)(x)

        x = UpSampling2D()(x)
        x = Conv2DTranspose(128, (3, 3), padding='same')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)

        x = UpSampling2D()(x)
        x = Conv2DTranspose(64, (3, 3), padding='same')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)

        x = UpSampling2D()(x)
        x = Conv2DTranspose(32, (3, 3), padding='same')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)

        x = UpSampling2D()(x)
        x = Conv2DTranspose(origin_channel, (3, 3), padding='same')(x)
        output = Activation('tanh')(x)

        return Model(inputs=fake_input, outputs=output, name='Generator')