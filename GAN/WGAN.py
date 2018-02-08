from keras.layers import *
from keras.models import *
from keras.optimizers import *
import keras.backend as K
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io,transform
from keras.initializers import RandomNormal
conv_init = RandomNormal(0, 0.02)

class WGAN():
    # loss_function:'gradient_penalty' or 'clip'
    def __init__(self, image_size=(64, 64, 1), random_vector_size=100, diters=5, D_lr=0.0002, G_lr=0.0002, loss_function='gradient_penalty', clamp=0.01, lamda=10):
        self.image_size = image_size
        self.random_vector_size = random_vector_size
        self.diters = diters
        self.loss_function = loss_function

        # discriminator
        self.discriminator = self.Discriminator()
        # generator
        self.generator = self.Generator()

        self.D_train, self.G_train, self.D_clamp = self.WGAN_train(loss_function, D_lr=D_lr, G_lr=G_lr, clamp=clamp,
                                                                   lamda=lamda)

    def WGAN_train(self, loss_function, D_lr, G_lr, clamp, lamda):
        assert loss_function=='gradient_penalty' or loss_function=='clip'

        x_real = Input(shape=self.image_size)
        fake_vectors = Input(shape=(self.random_vector_size,))
        x_fake = self.generator(fake_vectors)
        loss_real = K.mean(self.discriminator(x_real))
        loss_fake = K.mean(self.discriminator(x_fake))

        # loss for generator
        loss = -loss_fake
        training_updates = RMSprop(lr=G_lr).get_updates(loss, self.generator.trainable_weights)
        G_train = K.function([fake_vectors], [loss], training_updates)

        # clip step
        if loss_function == 'clip':
            # loss for discriminator
            loss = loss_fake - loss_real
            training_updates = RMSprop(lr=D_lr).get_updates(loss, self.discriminator.trainable_weights)
            D_train = K.function([x_real, fake_vectors], [loss_real, loss_fake], training_updates)

            # clip
            clamp_lower, clamp_upper = clamp * -1., clamp
            weights_clip = [K.update(x, K.clip(x, clamp_lower, clamp_upper)) for x in self.discriminator.trainable_weights]
            D_clamp = K.function([], [], weights_clip)

            return D_train, G_train, D_clamp

        # gradient penalty step
        else:
            # loss for discriminator
            e = K.placeholder(shape=(None, 1, 1, 1))
            x_mixed = Input(shape=self.image_size, tensor=e * x_real + (1 - e) * x_fake)
            x_mixed_gradient = K.gradients(self.discriminator(x_mixed), [x_mixed])[0]
            x_mixed_gradient_norm = K.sqrt(K.sum(K.square(x_mixed_gradient), axis=[1, 2, 3]))  # not norm in batch_size
            gradient_penalty = K.mean(K.square(x_mixed_gradient_norm - 1))
            loss = loss_fake - loss_real + lamda * gradient_penalty
            training_updates = RMSprop(lr=D_lr).get_updates(loss, self.discriminator.trainable_weights)
            D_train = K.function([x_real, fake_vectors, e], [loss_real, loss_fake], training_updates)

            return D_train, G_train, None

    def fit(self, path, batch_size=64, epochs=10000, draw=False, shuffle=True):
        train_list = np.array(os.listdir(path))
        train_index = np.arange(len(train_list))
        self.path = path
        self.batch_size = batch_size
        self.epochs = epochs
        gen_iterations = 0
        batches = len(train_index) // batch_size
        for i in range(epochs):
            loss_real_all=0
            loss_fake_all=0
            loss_D_all=0
            loss_G_all=0
            j = 0
            print('epochs:' + str(i + 1) + '/' + str(epochs))
            d_print = '\r[steps:%d/%d (diters:%d/%d)] loss_real: %.6f, loss_fake: %.6f, loss_D: %.6f' % (
                j, batches, 0, 0, loss_real_all, loss_fake_all, loss_D_all)
            g_print = '   [gen_iterations:%d] loss_G: %.6f' % (gen_iterations, loss_G_all)
            if shuffle:
                np.random.shuffle(train_index)
            while j < batches:
                if gen_iterations < 25 or gen_iterations % 200 == 0:
                    diters = 100
                else:
                    diters = self.diters
                q = 0
                # train discriminator
                while q < diters and j < batches:
                    temp_index = train_index[j * batch_size:min((j + 1) * batch_size, len(train_index) - 1)]
                    j += 1;q += 1
                    x_real = self.train_generator(train_list, temp_index)
                    fake_vectors = np.random.normal(size=[batch_size, self.random_vector_size])
                    if self.loss_function=='gradient_penalty':
                        e = np.random.uniform(size=(batch_size, 1, 1, 1))
                        loss_real, loss_fake = self.D_train([x_real,fake_vectors,e])
                    else:
                        self.D_clamp([])
                        loss_real, loss_fake = self.D_train([x_real, fake_vectors])
                    loss_real_all += loss_real
                    loss_fake_all += loss_fake
                    loss_D_all += (loss_fake - loss_real)
                    d_print = '\r[steps:%d/%d (diters:%d/%d)] loss_real: %.6f, loss_fake: %.6f, loss_D: %.6f' % (
                        j, batches, q, diters, loss_real_all, loss_fake_all, loss_D_all)
                    print(d_print+g_print, end='      ', flush=True)

                # plot fake image
                if draw and gen_iterations % 200 == 0:
                    self.plot_images(path='fake',step=gen_iterations)

                # train generator
                gen_iterations += 1
                fake_vectors = np.random.normal(size=[batch_size, self.random_vector_size])
                loss_G = self.G_train([fake_vectors])
                loss_G_all += loss_G
                g_print = '   [gen_iterations:%d] loss_G: %.6f' % (gen_iterations, loss_G_all)
                print(d_print + g_print, end='      ', flush=True)
            print('\n')
        return self

    def train_generator(self,train_list,temp_index):
        train_list_batch=train_list[temp_index]
        images_train=[]
        for s in train_list_batch:
            img = io.imread(self.path + "/" + s, as_grey=True)
            img = transform.resize(img, (self.image_size[0], self.image_size[1]), mode='reflect')
            img *= 255.
            img = self.preprocess_input(img)
            images_train.append(img)
        images_train=np.reshape(images_train, (self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2]))
        return images_train

    def preprocess_input(self,x):
        x = x / 255.
        x = x * 2.
        x = x - 1.
        return x

    def preprocess_output(self, x):
        x = x + 1.
        x = x / 2.
        x = x * 255.
        return x

    def plot_images(self, samples=25, noise=None, step=0, image_size=(64, 64, 1), path=None,show=False):
        if noise is None:
            noise = np.random.normal(size=[samples, self.random_vector_size])
        filename = "fake_epoches" + str(step) + ".png"
        images = self.preprocess_output(self.generator.predict(noise))
        plt.figure(figsize=(12, 12))
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

        # 32*32
        x = Conv2D(32, (3, 3), strides=2, padding='same', kernel_initializer=conv_init, use_bias=False)(input)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = LeakyReLU(alpha=0.2)(x)

        # 16*16
        x = Conv2D(64, (3, 3), strides=2, padding='same', kernel_initializer=conv_init, use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = LeakyReLU(alpha=0.2)(x)

        # 8*8
        x = Conv2D(128, (3, 3), strides=2, padding='same', kernel_initializer=conv_init, use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = LeakyReLU(alpha=0.2)(x)

        # 4*4
        x = Conv2D(256, (3, 3), strides=2, padding='same', kernel_initializer=conv_init, use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = LeakyReLU(alpha=0.2)(x)

        # 4*4*1 conv filters with valid padding to get 1-d output, which is used to instead of the sigmoid
        x = Conv2D(1, (4, 4), strides=1, padding='valid', kernel_initializer=conv_init, use_bias=False)(x)
        output = Flatten()(x)

        return Model(inputs=input, outputs=output, name='Discriminator')

    def Generator(self):
        origin_channel = self.image_size[2]
        fake_input = Input(shape=(self.random_vector_size,))
        x = Reshape((1, 1, self.random_vector_size))(fake_input)

        # 4*4
        x = Conv2DTranspose(256, (4, 4), strides=1, padding='valid', kernel_initializer=conv_init, use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation('relu')(x)

        # 8*8
        x = Conv2DTranspose(128, (3, 3), strides=2, padding='same', kernel_initializer=conv_init, use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation('relu')(x)

        # 16*16
        x = Conv2DTranspose(64, (3, 3), strides=2, padding='same', kernel_initializer=conv_init, use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation('relu')(x)

        # 32*32
        x = Conv2DTranspose(32, (3, 3), strides=2, padding='same', kernel_initializer=conv_init, use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation('relu')(x)

        # 64*64
        x = Conv2DTranspose(origin_channel, (3, 3), strides=2, padding='same', kernel_initializer=conv_init, use_bias=False)(x)
        output = Activation('tanh')(x)

        return Model(inputs=fake_input, outputs=output, name='Generator')