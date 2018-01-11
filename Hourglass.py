from keras.layers import *
from keras.models import *
from keras import layers
from keras.optimizers import *
import numpy as np
from skimage import io

# heatmaps
# xrange 有效像素点范围
def get_heatmaps(labels, size=(64, 64), xrange=1):
    life_x = labels[0:5]
    life_y = labels[5:10]
    int_x = labels[10:15]
    int_y = labels[15:20]
    aff_x = labels[20:25]
    aff_y = labels[25:30]
    finger_x = labels[30:35]
    finger_y = labels[35:40]
    x_all=np.concatenate((life_x,int_x,aff_x,finger_x))
    y_all=np.concatenate((life_y,int_y,aff_y,finger_y))
    heatmaps=np.zeros((size[0],size[0],len(x_all)))
    for i in range(len(x_all)):
        heatmap=np.zeros(size)
        x=int(x_all[i]*size[0])
        y=int(y_all[i]*size[1])
        heatmap[max(0,x-xrange):min(size[0],(x+xrange+1)),max(0,y-xrange):min(size[1],(y+xrange+1))]=1
        heatmaps[:,:,i]=heatmap

    return heatmaps

# generator to heatmaps
def generator2heatmaps(file_list, label_list, batch_size, shuffle=True, random_seed=None):
    while True:
        if shuffle:
            if random_seed != None:
                random_seed += 1
                np.random.seed(random_seed)
            index = np.arange(file_list.shape[0])
            np.random.shuffle(index)
            file_list = file_list[index]
            label_list = label_list[index]
        count = 0
        x, y = [], []
        for i, path in enumerate(file_list):
            img = io.imread(path)
            img = np.array(img)
            x_temp = img
            y_temp = get_heatmaps(label_list[i, :])
            count += 1
            x.append(x_temp)
            y.append(y_temp)
            if count % batch_size == 0 and count != 0:
                x = np.array(x)
                x = x.reshape(batch_size, 128, 128, 1).astype("float32")
                y = np.array(y)
                yield x, [y, y, y, y, y, y]
                x, y = [], []

def preprocess_numpy_input(x):
    x /= 127.5
    x -= 1.
    return x

def Residual(x, filters):
    # Skip layer
    shortcut = Conv2D(filters, (1, 1), padding='same')(x)

    # Residual block
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(int(filters / 2), (1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(int(filters / 2), (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same', use_bias=False)(x)
    x = layers.add([x, shortcut])

    return x

def Hourglass(x, level, filters):
    # up layer
    up = Residual(x, filters)

    # low layer
    low = MaxPooling2D()(x)
    low = Residual(low, filters)
    if level>1:
        low = Hourglass(low, level-1, filters)
    else:
        low = Residual(low, filters)
    low = Residual(low, filters)
    low = UpSampling2D()(low)
    x = layers.add([up, low])

    return x

def model(input_shape=(128, 128, 1), labels=40, nstack=6, level=4, filters=256, preprocess=True):
    img_input = Input(shape=input_shape)

    if preprocess:
        x = Lambda(preprocess_numpy_input)(img_input)
    else:
        x = img_input

    # 128*128
    x = Conv2D(64, (7, 7), strides=2, padding='same', use_bias=False)(x)
    # 64*64
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Residual(x, int(filters/2))
    x = MaxPooling2D()(x)
    # 32*32
    x = Residual(x, int(filters/2))
    middle_x = Residual(x, filters)
    outputs=[]

    for i in range(nstack):
        x = Hourglass(middle_x, level, filters)
        x = Residual(x, filters)
        x = Conv2D(filters, (1, 1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        temp_output = Conv2D(labels, (1, 1), padding='same', name='nstack_'+str(i+1))(x)
        outputs.append(temp_output)

        if i < nstack-1:
            x = Conv2D(filters, (1, 1), padding='same')(x)
            temp_output_ = Conv2D(filters, (1, 1), padding='same')(temp_output)
            middle_x = layers.add([middle_x, x, temp_output_])

    # Create model.
    model = Model(img_input, outputs, name='hourglass')

    return model

# test test
model=model((64,64,3),labels=10,preprocess=False)
optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mse', optimizer=optimizer)
X_train=np.random.random((1000,64,64,3))
y=np.random.random((1000,16,16,10))
model.fit(X_train,[y,y,y,y,y,y],verbose=1)