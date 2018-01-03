from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
import os
import numpy as np
from skimage import io,transform
import matplotlib.pyplot as plt

class DCGAN():
    def __init__(self,image_size=(64,64,3),random_vector_size=100):
        self.image_size=image_size
        self.random_vector_size=random_vector_size

        # adversarial
        self.discriminator=self.Discriminator()
        self.generator=self.Generator()
        self.adversarial = self.Adversarial(self.discriminator,self.generator)
        self.optimizer = Adam(lr=0.0002,beta_1=0.5,beta_2=0.999)
        self.adversarial.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        self.check_point = ModelCheckpoint('GAN_model.h5', monitor='loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    def fit(self,path,batch_size=64,epochs=3000,draw=False):
        train_list=np.array(os.listdir(path))
        self.path=path
        self.steps=0
        self.adversarial.fit_generator(generator=self.generate_for_train(train_list,batch_size,random_seed=2000,draw=draw),steps_per_epoch=int(len(train_list)/batch_size),
                                       epochs=epochs,verbose=1,callbacks=[self.check_point])

        return self

    # 生成器generator for GAN with shuffle
    def generate_for_train(self,file_list, batch_size, shuffle=True, random_seed=None, draw=True):
        while True:
            # 洗牌
            if shuffle:
                if random_seed != None:
                    random_seed += 1
                    np.random.seed(random_seed)
                index = np.arange(file_list.shape[0])
                np.random.shuffle(index)
                file_list = file_list[index]
            count = 0
            x = []
            for i, path in enumerate(file_list):
                img = io.imread(self.path + "/" + path)
                img = np.array(img)
                img = transform.resize(img,(self.image_size[0],self.image_size[1],self.image_size[2]))
                x_temp = self.preprocess_input(img)
                count += 1
                x.append(x_temp)
                if count % batch_size == 0 and count != 0:
                    x = np.array(x)
                    x = x.reshape(batch_size, self.image_size[0], self.image_size[1], self.image_size[2]).astype("float32")
                    noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
                    images_fake = self.generator.predict(noise)
                    x_D = np.concatenate((x, images_fake), axis=0)
                    y_D = np.ones(2 * batch_size)
                    y_D[batch_size:] = 0
                    x_G = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
                    y_G = np.ones(batch_size)
                    yield [x_D, x_G], [y_D, y_G]
                    x = []
            self.steps+=1
            if draw:
                self.plot_images(path='fake', step=self.steps)

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

    def plot_images(self, samples=16, noise=None, step=0, image_size=(64, 64, 3), path=None,show=False):
        if noise is None:
            noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
        filename = "fake_epoch" + str(step) + ".png"
        images = self.preprocess_output(self.generator.predict(noise))
        plt.figure(figsize=(6, 6))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i + 1)
            image = images[i, :, :, :]
            image = np.reshape(image, [image_size[0],image_size[1], image_size[2]])
            plt.imshow(image)
            plt.axis('off')
        plt.tight_layout()
        if path != None:
            filename = path + '/' + filename
        if show:
            plt.show()
        plt.savefig(filename)
        plt.close('all')

    def Adversarial(self,discriminator,generator):
        input_D = Input(shape=(self.image_size[0], self.image_size[1], self.image_size[2]))
        output_D = discriminator(input_D)

        input_G = Input(shape=(self.random_vector_size,))
        output_G = generator(input_G)
        output_GD = discriminator(output_G)

        return Model(inputs=[input_D,input_G], outputs=[output_D,output_GD], name='Adversarial')

    def Discriminator(self):
        input = Input(shape=(self.image_size[0], self.image_size[1], self.image_size[2]))
        x = Conv2D(64, (3, 3), strides=2, padding='same')(input)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)

        x = Conv2D(128, (3, 3), strides=2, padding='same')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)

        x = Conv2D(256, (3, 3), strides=2, padding='same')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)

        x = Conv2D(512, (3, 3), strides=2, padding='same')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)

        x = GlobalAveragePooling2D()(x)
        output = Dense(1, activation='sigmoid')(x)

        return Model(inputs=input, outputs=output, name='Discriminator')

    def Generator(self):
        dim = int(self.image_size[0] / 16)
        origin_channel = self.image_size[2]
        fake_input = Input(shape=(self.random_vector_size,))
        x = Dense(dim * dim * 512)(fake_input)
        x = Activation('relu')(x)
        x = Reshape((dim, dim, 512))(x)

        x = UpSampling2D()(x)
        x = Conv2DTranspose(256, (3, 3), padding='same')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)

        x = UpSampling2D()(x)
        x = Conv2DTranspose(128, (3, 3), padding='same')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)

        x = UpSampling2D()(x)
        x = Conv2DTranspose(64, (3, 3), padding='same')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)

        x = UpSampling2D()(x)
        x = Conv2DTranspose(origin_channel, (3, 3), padding='same')(x)
        output = Activation('tanh')(x)

        return Model(inputs=fake_input, outputs=output, name='Generator')