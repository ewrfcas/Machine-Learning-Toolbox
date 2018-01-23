from keras import backend as K

def focal_loss(y_true, y_pred, gamma = 2, n_class=5):
    return K.mean(K.pow(K.ones(shape=(n_class,))-y_pred,gamma))*K.categorical_crossentropy(y_pred,y_true)
