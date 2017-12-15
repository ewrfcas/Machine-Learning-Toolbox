from keras import backend as K

#TPR
def true_positive_rate(y_true, y_pred,mode='p'):
    threshold_value = 0.5
    if mode=='n':
        threshold_value=1-threshold_value
    # works as round() with threshold_value
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
    true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
    real_positives = K.sum(K.clip(y_true, 0, 1))
    return true_positives / (real_positives + K.epsilon())

#TNR
def true_negative_rate(y_true, y_pred):
    y_true = K.ones((32,))-y_true
    y_pred = K.ones((32,))-y_pred
    return true_positive_rate(y_true, y_pred,mode='n')

#G-MEAN:(TPR*TNR)^0.5
def geometric_mean(y_true, y_pred):
    return K.sqrt(true_positive_rate(y_true, y_pred)*true_negative_rate(y_true, y_pred))

#M-ACC (TPR*TNR)/2
def mean_accuracy(y_true, y_pred):
    return (true_positive_rate(y_true, y_pred)+true_negative_rate(y_true, y_pred))/2