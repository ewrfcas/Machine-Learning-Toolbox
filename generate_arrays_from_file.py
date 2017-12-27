import numpy as np
from keras import utils
import os
import h5py
from skimage import io
def generate_arrays_from_file(path):
    while True:
        f = open(path)
        batch_size=64
        count=0
        x=[]
        y=[]
        for line in f:
            xs = line.split(',')
            xs=list(map(lambda x:float(x),xs))
            x_temp=np.array(xs[0:-1])
            y_temp=int(xs[-1])
            count+=1
            x.append(x_temp)
            y.append(y_temp)
            if count==batch_size:
                x=np.concatenate(x,axis=0)
                x=np.reshape(x,(batch_size,60,40,1))
                y=utils.to_categorical(np.array(y),5)
                yield x,y
                x = []
                y = []
                count=0
        f.close()

#for autoencoder
def generate_pics_from_file(path):
    while True:
        batch_size=64
        count=0
        x=[]
        for pic in os.listdir(path):
            img = io.imread(path+"/"+pic)
            img= cv.resize(img,(32,32))
            x.append(img)
            count+=1
            if count==batch_size:
                x=np.array(x)
                x = np.reshape(x,(batch_size, 32, 32, 3))
                yield x, x
                x = []
                count = 0

def generate_for_lung(file_list, label_list, batch_size):
    while True:
        count = 0
        x = []
        y = []
        for i,path in enumerate(file_list):
            file=h5py.File(path, 'r')
            x_temp = np.array(file['data'])
            x_temp = np.resize(x_temp,(128,128,128))
            file.close()
            y_temp = int(label_list[i])
            count += 1
            x.append(x_temp)
            y.append(y_temp)
            if count == batch_size:
                x = np.concatenate(x)
                x = np.reshape(x,(batch_size,128,128,128))
                y = np.array(y)
                yield x, y
                x = []
                y = []
                count = 0

#生成器generator for image with shuffle
def generate_for_kp(file_list, label_list, batch_size, shuffle=True, random_seed=None):
    while True:
        #洗牌
        if shuffle:
            if random_seed!=None:
                np.random.seed(random_seed)
            index=np.arange(file_list.shape[0])
            np.random.shuffle(index)
            file_list=file_list[index]
            label_list=label_list[index]
        count = 0
        x, y = [], []
        for i,path in enumerate(file_list):
            img=io.imread(path)
            img = np.array(img)
            x_temp=img/255.0
            y_temp=label_list[i,:]
            count += 1
            x.append(x_temp)
            y.append(y_temp)
            if count % batch_size == 0 and count != 0:
                x = np.array(x)
                x = x.reshape(batch_size, 96, 96, 3).astype("float32")
                y = np.array(y)
                yield x, y
                x, y = [], []


def generate_for_kp_test(file_list, batch_size):
    while True:
        count = 0
        x = []
        for path in file_list:
            img = io.imread(path)
            img = np.array(img)
            x_temp = img / 255.0
            count += 1
            x.append(x_temp)
            if count % batch_size == 0 and count != 0:
                x = np.array(x)
                x = x.reshape(batch_size, 96, 96, 1).astype("float32")
                yield x
                x = []