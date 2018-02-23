import numpy as np
import os
from keras.layers import *
from keras.models import *
from keras.optimizers import *
import keras.backend as K
import matplotlib.pyplot as plt
from skimage import io,transform
from keras.initializers import RandomNormal
# plt.switch_backend('agg')
# conv_init = RandomNormal(0, 0.02)

class StarGAN():
    def __init__(self, image_size=(128, 128, 3), n_class=5, repeat_num=6, diters=5, D_lr=0.0002, G_lr=0.0002, lamda_gp=10, lamda_cls=1, lamda_rec=10):
        self.image_size = image_size
        self.repeat_num = repeat_num
        self.diters = diters
        self.n_class = n_class
        self.D_lr = D_lr
        self.G_lr = G_lr
        self.lamda_gp = lamda_gp
        self.lamda_cls = lamda_cls
        self.lamda_rec = lamda_rec

        # discriminator
        self.discriminator = self.Discriminator()
        # generator
        self.generator = self.Generator()

    def save_weights(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        self.discriminator.save_weights(path + '/discriminator_weights.h5')
        self.generator.save_weights(path + '/generator_weights.h5')

    def load_weights(self, path):
        self.discriminator.load_weights(path + '/discriminator_weights.h5')
        self.generator.load_weights(path + '/generator_weights.h5')

    def starGAN_train(self, D_lr, G_lr, lamda_gp, lamda_cls, lamda_rec):

        x_real = Input(shape=self.image_size)
        label_real = Input(shape=(self.n_class,))
        label_fake = Input(shape=(self.n_class,))
        label_real_matrix = Input(shape=(self.image_size[0],self.image_size[1],self.n_class))
        label_fake_matrix = Input(shape=(self.image_size[0],self.image_size[1],self.n_class))
        x_fake = self.generator([x_real, label_fake_matrix])

        # loss for discriminator
        d_out_src_real, d_out_cls_real = self.discriminator(x_real)
        d_loss_real = -K.mean(d_out_src_real)
        d_loss_cls = K.mean(K.categorical_crossentropy(label_real, d_out_cls_real))
        # cal acc
        label_pred = K.cast(K.greater(K.clip(d_out_cls_real, 0, 1), 0.5), K.floatx())
        d_acc = 1 - K.mean(K.clip(K.sum(K.abs(label_real - label_pred), axis=1), 0, 1))
        d_out_src_fake, d_out_cls_fake = self.discriminator(x_fake)
        d_loss_fake = K.mean(d_out_src_fake)

        # gradient penalty
        e = K.placeholder(shape=(None, 1, 1, 1))
        x_mixed = Input(shape=self.image_size, tensor=e * x_real + (1 - e) * x_fake)
        x_mixed_gradient = K.gradients(self.discriminator(x_mixed), [x_mixed])[0]
        x_mixed_gradient_norm = K.sqrt(K.sum(K.square(x_mixed_gradient), axis=[1, 2, 3]))  # not norm in batch_size
        gradient_penalty = K.mean(K.square(x_mixed_gradient_norm - 1))

        d_loss = d_loss_real + d_loss_fake + lamda_gp * gradient_penalty + lamda_cls * d_loss_cls
        d_training_updates = RMSprop(lr=D_lr).get_updates(d_loss, self.discriminator.trainable_weights)
        D_train = K.function([x_real, label_real, label_real_matrix, label_fake, label_fake_matrix, e], [d_loss, d_acc], d_training_updates)

        # loss for generator
        x_rec = self.generator([x_fake, label_real_matrix])
        g_out_src_fake, g_out_cls_fake = self.discriminator(x_fake)
        g_loss_fake = -K.mean(g_out_src_fake)
        g_loss_rec = K.mean(K.abs(x_real - x_rec))
        g_loss_cls = K.mean(K.categorical_crossentropy(label_fake, g_out_cls_fake))

        g_loss = g_loss_fake + lamda_rec * g_loss_rec + lamda_cls * g_loss_cls
        g_training_updates = RMSprop(lr=G_lr).get_updates(g_loss, self.generator.trainable_weights)
        G_train = K.function([x_real, label_real, label_real_matrix, label_fake, label_fake_matrix], [g_loss], g_training_updates)

        return D_train, G_train

    def fit(self, X_path, y, batch_size=16, epochs=20, draw=False, shuffle=True):
        # X_path:path array[string], y:one-hot array[array[int]]
        train_index = np.arange(len(X_path))
        self.batch_size = batch_size
        self.epochs = epochs
        gen_iterations = 0
        batches = len(train_index) // batch_size
        D_train, G_train = self.starGAN_train(self.D_lr, self.G_lr, self.lamda_gp, self.lamda_cls, self.lamda_rec)
        for i in range(epochs):
            # train step
            if epochs-i<10:
                D_train, G_train = self.starGAN_train(self.D_lr*((epochs-i)/10), self.G_lr*((epochs-i)/10), self.lamda_gp, self.lamda_cls,
                                                      self.lamda_rec)
            d_loss_all = 0
            d_acc_all = 0
            g_loss_all = 0
            j = 0
            g = 0
            print('epochs:' + str(i + 1) + '/' + str(epochs))
            d_print = '\r[steps:%d/%d (diters:%d/%d)] d_loss: %.6f, d_acc: %.4f' % (j, batches, 0, 0, d_loss_all, d_acc_all)
            g_print = '   [gen_iterations:%d] g_loss: %.6f' % (gen_iterations, g_loss_all)
            if shuffle:
                np.random.shuffle(train_index)
            while j < batches:
                q = 0
                # train discriminator
                while q < self.diters and j < batches:
                    temp_index = train_index[j * batch_size:min((j + 1) * batch_size, len(train_index) - 1)]
                    j += 1;q += 1
                    x_real,label_real,label_fake = self.train_generator(X_path, y, temp_index)
                    label_real_matrix=self.label2matrix(label_real)
                    label_fake_matrix=self.label2matrix(label_fake)
                    e = np.random.uniform(size=(batch_size, 1, 1, 1))
                    [d_loss,d_acc] = D_train([x_real, label_real, label_real_matrix, label_fake, label_fake_matrix, e])
                    d_loss_all += d_loss
                    d_acc_all += d_acc
                    d_print = '\r[steps:%d/%d (diters:%d/%d)] d_loss: %.6f, d_acc: %.4f' % (j, batches, q, self.diters, d_loss_all/j, d_acc_all/j)
                    print(d_print+g_print, end='      ', flush=True)

                # plot fake image
                if draw and gen_iterations % 200 == 0:
                    self.plot_images(X_path=X_path, y=y, path='fake',step=gen_iterations)

                # train generator
                gen_iterations += 1
                g += 1
                [g_loss] = G_train([x_real, label_real, label_real_matrix, label_fake, label_fake_matrix])
                g_loss_all += g_loss
                g_print = '   [gen_iterations:%d] g_loss: %.6f' % (gen_iterations, g_loss_all/g)
                print(d_print + g_print, end='      ', flush=True)
            print('\n')
        return self

    def train_generator(self, X_path, y, temp_index, enhance=False):
        X_path=np.array(X_path)
        train_list_batch=X_path[temp_index]
        label_real=y[temp_index,:]
        images_train=[]
        for s in train_list_batch:
            img = io.imread(s)
            img = self.preprocess_input(img)
            images_train.append(img)
        images_train=np.reshape(images_train, (len(temp_index), self.image_size[0], self.image_size[1], self.image_size[2]))

        # get label_fake
        random_index=np.random.choice(np.arange(y.shape[0]),len(temp_index))
        label_fake=y[random_index,:]

        return images_train,label_real,label_fake

    def label2matrix(self,y):
        # label转矩阵 batch*n_class->batch*n_class*w*h，如[0,1,0]转为3个h*w矩阵——全零矩阵[0],全一矩阵[1],全零矩阵[0]
        y_matrix=np.zeros((y.shape[0],self.image_size[0],self.image_size[1],y.shape[1]))
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if y[i,j]==1:
                    y_matrix[i,:,:,j]=1
        return y_matrix

    def preprocess_input(self,x):
        x = x / 255.
        x = x * 2.
        x = x - 1.
        return x

    def preprocess_output(self, x):
        x = x + 1.
        x = x / 2.
        # x = x * 255.
        return x

    def plot_images(self, X_path, y, samples=5, step=0, path=None,show=False):
        filename = "gen_steps" + str(step) + ".png"
        temp_index = np.random.choice(np.arange(y.shape[0]), samples, replace=False)
        x_real, label_real, _ = self.train_generator(X_path, y, temp_index)
        label_blond_hair = label_real
        label_blond_hair[:,0]=0
        label_blond_hair[:,1]=1
        label_blond_hair[:,2]=0
        label_male=label_real
        label_male[:,3]=1
        label_old=label_real
        label_old[:,4]=0
        # generate fake image
        x_blond_hair=self.generator.predict([x_real,self.label2matrix(label_blond_hair)])
        x_male=self.generator.predict([x_real,self.label2matrix(label_male)])
        x_old=self.generator.predict([x_real,self.label2matrix(label_old)])

        plt.figure(figsize=(10, 10))
        for i in range(x_real.shape[0]):
            #plot origin
            plt.subplot(samples, 4, i * 4 + 1)
            plt.imshow(self.preprocess_output(x_real[i, :, :, :]))
            plt.title('origin')
            plt.axis('off')

            plt.subplot(samples, 4, i * 4 + 2)
            plt.imshow(self.preprocess_output(x_blond_hair[i, :, :, :]))
            plt.title('blond hair')
            plt.axis('off')

            plt.subplot(samples, 4, i * 4 + 3)
            plt.imshow(self.preprocess_output(x_male[i, :, :, :]))
            plt.title('male')
            plt.axis('off')

            plt.subplot(samples, 4, i * 4 + 4)
            plt.imshow(self.preprocess_output(x_old[i, :, :, :]))
            plt.title('old')
            plt.axis('off')
        plt.tight_layout()
        if path != None:
            filename = path + '/' + filename
        if show:
            plt.show()
        plt.savefig(filename)
        plt.close('all')

    def ResidualBlock(self, x, filters):
        # Skip layer
        shortcut = Conv2D(filters, (1, 1), padding='same')(x)

        # Residual block
        x = Conv2D(filters, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x, training=1)
        x = Activation('relu')(x)
        x = Conv2D(filters, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x, training=1)
        x = add([x, shortcut])

        return x

    def Discriminator(self):
        input = Input(shape=(self.image_size[0], self.image_size[1], self.image_size[2]))

        x = Conv2D(64, (4, 4), strides=2, padding='same')(input)
        x = LeakyReLU(alpha=0.01)(x)

        k_size=self.image_size[0]//2
        filters=128
        for i in range(self.repeat_num-3):
            x = Conv2D(filters, (3, 3), strides=2, padding='same')(x)
            filters*=2
            k_size//=2
            x = LeakyReLU(alpha=0.2)(x)

        # k_size*k_size*1 conv filters with valid padding to get 1-d output, which is used to instead of the sigmoid
        output_x = Conv2D(1, (k_size, k_size), strides=1, padding='valid', use_bias=False)(x)
        output_y = Conv2D(self.n_class, (k_size, k_size), strides=1, padding='valid', use_bias=False)(x)
        output_x = Flatten()(output_x)
        output_y = Flatten()(output_y)

        return Model(inputs=input, outputs=[output_x,output_y], name='Discriminator')

    def Generator(self):
        origin_channel = self.image_size[2]
        x_real = Input(shape=(self.image_size[0], self.image_size[1], origin_channel))
        label_fake = Input(shape=(self.image_size[0], self.image_size[1], self.n_class))
        x = Concatenate(axis=-1)([x_real,label_fake])

        x = Conv2D(64, (5, 5), strides=1, padding='same', use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x, training=1)
        x = Activation('relu')(x)

        # down-sampling
        x = Conv2D(128, (4, 4), strides=2, padding='same', use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x, training=1)
        x = Activation('relu')(x)

        x = Conv2D(256, (4, 4), strides=2, padding='same', use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x, training=1)
        x = Activation('relu')(x)

        # Bottleneck
        for i in range(self.repeat_num):
            x = self.ResidualBlock(x, 512)

        # up-sampling
        x = Conv2DTranspose(256, (4, 4), strides=2, padding='same', use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x, training=1)
        x = Activation('relu')(x)

        x = Conv2DTranspose(128, (4, 4), strides=2, padding='same', use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x, training=1)
        x = Activation('relu')(x)

        x = Conv2D(origin_channel, (5, 5), strides=1, padding='same', use_bias=False)(x)
        output = Activation('tanh')(x)

        return Model(inputs=[x_real,label_fake], outputs=output, name='Generator')