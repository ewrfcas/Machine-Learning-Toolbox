from keras.layers import *

# focal loss
# (1-y)^k*log(y)
# 交叉熵前乘以1-yred指数，使得低置信度的loss更高，高置信度的loss更低
def focal_loss(y_true, y_pred, gamma=2, n_class=5):
    return K.mean(K.pow(K.ones(shape=(n_class,))-y_pred,gamma))*K.categorical_crossentropy(y_pred,y_true)