import numpy as np
from keras import utils
import os
import h5py
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
            img = cv.imread(path+"/"+pic)
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
            x_temp = file['data']
            file.close()
            y_temp = int(label_list[i])
            count += 1
            x.append(x_temp)
            y.append(y_temp)
            if count == batch_size:
                x = np.concatenate(x)
                x = np.resize(x,(batch_size,128,128,128))
                y = np.array(y)
                yield x, y
                x = []
                y = []
                count = 0