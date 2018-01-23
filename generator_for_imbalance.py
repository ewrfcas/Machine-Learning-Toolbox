from skimage import io,transform
import numpy as np
import keras

#生成器generator for imbalanced data，每个batch中样本等量
def generator_train(file_list, label_list, batch_size, shuffle=True, random_seed=None, imbalance=True):
    while True:
        if shuffle:
            if random_seed!=None:
                random_seed+=1
                np.random.seed(random_seed)
            index=np.arange(file_list.shape[0])
            np.random.shuffle(index)
            file_list=file_list[index]
            label_list=label_list[index]
        count = 0
        label_true=keras.utils.to_categorical(np.array(label_list))
        x, y = [], []
        if imbalance:#batch内样本不同类别平衡
            labels=np.unique(label_list)
            index_list=[]
            max_num=0
            for l in labels:
                index_list.append(np.where(label_list==l)[0])
                if len(np.where(label_list==l)[0])>max_num:
                    max_num=len(np.where(label_list==l)[0])
            for i in range(max_num):#i保持循环到最多的类结束为止
                for j in range(len(index_list)):#不同类别各取一个
                    if i%len(index_list[j])==0 and i!=0:#当这个类别已经循环了一遍了
                        np.random.shuffle(index_list[j])#需要洗牌这个类别
                    x_temp=io.imread(file_list[index_list[j][i%len(index_list[j])]])
                    y_temp=label_true[index_list[j][i%len(index_list[j])],:]
                    count+=1
                    x.append(x_temp)
                    y.append(y_temp)
                    if count % batch_size == 0 and count != 0:
                        x = np.array(x)
                        x = x.reshape(batch_size, 128, 128, 3)
                        y = np.array(y)
                        yield x, y
                        x, y = [], []
        else:#for validation
            for i,path in enumerate(file_list):
                x_temp = io.imread(path)
                y_temp=label_true[i,:]
                count += 1
                x.append(x_temp)
                y.append(y_temp)
                if count % batch_size == 0 and count != 0:
                    x = np.array(x)
                    x = x.reshape(batch_size, 128, 128, 3)
                    y = np.array(y)
                    yield x, y
                    x, y = [], []