import numpy as np

#binary
#ratio: majority'=(majority-minority)*ratio+minoirty
def RandomUnderSampling(X,y,ratio=0,random_seed=2,replace=False):
    np.random.seed(random_seed)
    labels=np.unique(y)
    if len(np.where(y==labels[0])[0])>=len(np.where(y==labels[1])[0]):
        majority_label=labels[0]
        majority_num=len(np.where(y==labels[0])[0])
        minority_label=labels[1]
        minority_num=y.shape[0]-majority_num
    else:
        majority_label=labels[1]
        majority_num=len(np.where(y==labels[1])[0])
        minority_label=labels[0]
        minority_num=y.shape[0]-majority_num
    majority_X=X[y==majority_label,:]
    minority_X=X[y==minority_label,:]
    random_index=np.random.choice(np.arange(majority_num),minority_num+(majority_num-minority_num)*ratio,replace=replace)
    majority_X=majority_X[random_index,:]
    final_X=np.concatenate((minority_X,majority_X),axis=0)
    final_y=np.ones(final_X.shape[0])
    final_y[0:minority_num]=final_y[0:minority_num] * minority_label
    final_y[minority_num:] = final_y[minority_num:] * majority_label

    return final_X,final_y