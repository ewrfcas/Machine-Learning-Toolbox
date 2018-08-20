from DeMeshNet import DeMeshNet
from skimage import io, transform
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
# model = DeMeshNet(model_folder='model/Hourglass_modelsV3_epoch7.h5',gpu_use=0.3,input_shape=(224, 176, 3), nstack=2, level=4, module=1, filters=128)

base_path='../../wangdexun/image_data/data_tang/person_img/'
path1=np.array(os.listdir(base_path))

# 分割path
split_num=12
temp_len=len(path1)//split_num
base_paths=[]
for i in range(split_num):
    if i != split_num-1:
        base_paths.append(path1[i*temp_len:(i+1)*temp_len])
    else:
        base_paths.append(path1[i*temp_len:])

def demesh(base_path, path1):
    model = DeMeshNet(model_folder='model/Hourglass_modelsV3_epoch7.h5',gpu_use=0.02,input_shape=(224, 176, 3), nstack=2, level=4, module=1, filters=128)
    mis_num=0
    for p in tqdm(path1):
        path2=base_path+p
        for p2 in os.listdir(path2):
            if '_m' in p2 and '_d' not in p2:
                try:
                    img_demeshed=model.predict_one(path2+'/'+p2)
                    io.imsave((path2+'/'+p2).split('.jpg')[0]+'_d.jpg',img_demeshed)
                except:
                    mis_num+=1
    return mis_num

from multiprocessing import Pool
pool=Pool()
result=[]
for i in base_paths:
    result.append(pool.apply_async(demesh, kwds={'base_path':base_path, 'path1':i}))
pool.close()
pool.join()

mis_num=np.sum([i.get() for i in result])
print('wrong:',mis_num)
