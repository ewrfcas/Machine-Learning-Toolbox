import tensorflow as tf

def huber_loss(y_true, y_pred, delta=1.0):
    return tf.losses.huber_loss(y_true,y_pred,delta=delta)