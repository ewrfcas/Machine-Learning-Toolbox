import tensorflow as tf
from keras.layers import *
import keras.backend as K

sess = tf.InteractiveSession()

inputs = tf.placeholder(tf.float32, [None, 64, 512])
labels = tf.placeholder(tf.float32, [None, 1])

# rank pearson loss
def spearman_loss(y_true_rank, y_pred_rank, eps=1e-10):
    y_true_mean = K.mean(y_true_rank)
    y_pred_mean = K.mean(y_pred_rank)
    u1 = (y_true_rank - y_true_mean)
    u2 = (y_pred_rank - y_pred_mean)
    u=K.sum(tf.multiply(u1,u2))
    d=K.sqrt(K.sum(K.square(u1))*K.sum(K.square(u2)))
    rou=tf.div(u,d+eps)
    return 1.-rou

def model(inputs, unit=256):
    x = Masking(mask_value=0)(inputs)
    x = LSTM(unit, return_sequences=False)(x,training=1)
    x = Dense(1, activation='sigmoid')(x)
    x_temp = tf.reshape(tf.transpose(x),(-1,))
    x_temps=[]
    for i in range(16):
        x_temps.append(x_temp)
    x_temps=tf.stack(x_temps)
    eps=1e-10
    x1=x-x_temps+eps
    x2=tf.abs(x-x_temps)+eps
    x=tf.div(x1,x2)
    x=tf.reshape(tf.reduce_sum(x,axis=1),(-1,1))

    return x

preds=model(inputs)
loss=spearman_loss(labels,preds)
train_step = tf.train.AdamOptimizer(0.005).minimize(loss)
sess.run(tf.global_variables_initializer())


X=np.random.random((320,64,512))
y=np.random.random(320)
for i in range(20):
    print(i,':')
    feed_dict = {inputs: X[i*16:(i+1)*16,:,:], labels: np.reshape(np.argsort(y[i*16:(i+1)*16]).astype(np.float32),(-1,1))}
    loss_value,_,preds_value = sess.run([loss, train_step,preds], feed_dict=feed_dict)
    print(preds_value)
    print(loss_value)