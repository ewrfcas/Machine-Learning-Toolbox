from keras.layers import *
from keras.models import *
from keras import layers
from skimage import io

# heatmaps
def makeGaussian(height, width, sigma = 2, center=None):
    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)[:, np.newaxis]
    if center is None:
        x0 =  width // 2
        y0 = height // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / sigma**2)

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
        x=int(x_all[i]*size[1])
        y=int(y_all[i]*size[0])
        heatmap=makeGaussian(size[0], size[1], sigma = 2, center=[x,y])
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

def Hourglass(x, level, module, filters):
    # up layer
    for i in range(module):
        x = Residual(x, filters)
    up = x

    # low layer
    low = MaxPooling2D()(x)
    for i in range(module):
        low = Residual(low, filters)
    if level>1:
        low = Hourglass(low, level-1, module, filters)
    else:
        for i in range(module):
            low = Residual(low, filters)
    for i in range(module):
        low = Residual(low, filters)
    low = UpSampling2D()(low)
    x = layers.add([up, low])

    return x

def model(input_shape=(256, 256, 1), labels=20, nstack=6, level=4, module=1, filters=256, preprocess=True):
    img_input = Input(shape=input_shape)

    if preprocess:
        x = Lambda(preprocess_numpy_input)(img_input)
    else:
        x = img_input

    # 256*256
    x = Conv2D(64, (7, 7), strides=2, padding='same', use_bias=False)(x)
    # 128*128
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Residual(x, int(filters/2))
    x = MaxPooling2D()(x)
    # 64*64
    x = Residual(x, int(filters/2))
    middle_x = Residual(x, filters)
    outputs=[]

    for i in range(nstack):
        x = Hourglass(middle_x, level, module, filters)
        for j in range(module):
            x = Residual(x, filters)
        x = Conv2D(filters, (1, 1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        temp_output = Conv2D(labels, (1, 1), padding='same', name='nstack_'+str(i+1))(x)
        outputs.append(temp_output)

        if i < nstack-1:
            x = Conv2D(filters, (1, 1), padding='same')(x)
            temp_output = Conv2D(filters, (1, 1), padding='same')(temp_output)
            middle_x = layers.add([middle_x, x, temp_output])

    # Create model.
    model = Model(img_input, outputs, name='hourglass')

    return model