from keras.layers import *
from keras.models import *
from keras.optimizers import *
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io,transform

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

    def fit(self,path,batch_size=64,epochs=10000,draw=False,shuffle=True):
        train_list=np.array(os.listdir(path))
        train_index = np.arange(len(train_list))
        self.path=path
        self.batch_size=batch_size
        self.epochs=epochs
        for i in range(epochs):
            print('epochs:'+str(i + 1) + '/' + str(epochs))
            d_acc=0
            g_acc=0
            if shuffle:
                np.random.shuffle(train_index)
            for j in range(int(len(train_index)/batch_size)):
                temp_index=train_index[j*batch_size:min((j+1)*batch_size,len(train_index)-1)]
                images_train=self.train_generator(train_list,temp_index)
                noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
                images_fake = self.generator.predict(noise)
                x = np.concatenate((images_train, images_fake), axis=0)
                y = np.ones([2 * batch_size, 1])
                y[batch_size:, :] = 0
                d_loss_acc=self.discriminator.train_on_batch(x, y)
                y = np.ones([batch_size, 1])
                noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
                g_loss_acc=self.adversarial.train_on_batch(noise, y)
                d_acc+=d_loss_acc[1] * 100
                g_acc+=g_loss_acc[1] * 100
                print("\rsteps:%d/%d discriminnator_loss/acc:[%.8f,%.2f%%], generator_loss/acc:[%.8f,%.2f%%]" % (
                j + 1, int(len(train_index) / batch_size), d_loss_acc[0], d_acc/(j+1), g_loss_acc[0],
                g_acc/(j+1)), end='      ', flush=True)
            if draw and (i+1)%5==0:
                self.plot_images(path='fake',step=i+1)
            print('\n')
        return self

    def train_generator(self,train_list,temp_index):
        train_list_batch=train_list[temp_index]
        images_train=[]
        for s in train_list_batch:
            img = io.imread(self.path + "/" + s,as_grey=True)
            img = transform.resize(img, (self.image_size[0],self.image_size[1]),mode='reflect')
            img*=255.
            img=self.preprocess_input(img)
            images_train.append(img)
        images_train=np.reshape(images_train, (self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2]))
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

    def plot_images(self, samples=25, noise=None, step=0, image_size=(64, 64, 1), path=None,show=False):
        if noise is None:
            noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
        filename = "fake_epoches" + str(step) + ".png"
        images = self.preprocess_output(self.generator.predict(noise))
        plt.figure(figsize=(8, 8))
        for i in range(images.shape[0]):
            plt.subplot(5, 5, i + 1)
            image = images[i, :, :, :]
            if image_size[2]==1:
                image = np.reshape(image, [image_size[0], image_size[1]])
                plt.imshow(image, cmap='gray')
            else:
                image = np.reshape(image, [image_size[0],image_size[1],image_size[2]])
                plt.imshow(image)
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