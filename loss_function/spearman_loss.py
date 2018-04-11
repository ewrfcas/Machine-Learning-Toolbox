import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import keras.backend as K

# rank pearson loss
def pearson_loss(y_true_rank, y_pred_rank, eps=1e-10):
    y_true_mean = K.mean(y_true_rank)
    y_pred_mean = K.mean(y_pred_rank)
    u1 = (y_true_rank - y_true_mean)
    u2 = (y_pred_rank - y_pred_mean)
    u=K.sum(tf.multiply(u1,u2))
    d=K.sqrt(K.sum(K.square(u1))*K.sum(K.square(u2)))
    rou=tf.div(u,d+eps)
    return 1.-rou

# feature to rank
class SpearRank(Layer):
    def __init__(self, eps=1e-10, **kwargs):
        self.eps = eps
        super(SpearRank, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SpearRank, self).build(input_shape)

    def call(self, x, mask=None):
        x_transpose = tf.reshape(tf.transpose(x), (1, -1))
        x1 = x - x_transpose
        x2 = tf.abs(x - x_transpose) + self.eps
        x = tf.div(x1, x2)
        x = tf.reshape(tf.reduce_sum(x, axis=1), (-1, 1))
        x = x + tf.reduce_min(x) * (-1)

        return x

def model(timesteps=64, dim=512, unit=256,ac='sigmoid', eps=1e-10):
    inputs = Input((timesteps, dim))
    x = Masking(mask_value=0)(inputs)
    x = LSTM(unit, return_sequences=False)(x)
    x = Dense(1, activation=ac)(x)
    x = SpearRank()(x)

    return Model(inputs=inputs, outputs=x)

model=model()
model.compile(optimizer='adam',loss=pearson_loss)
X=np.random.random((320,64,512))
y=np.random.random(320)
model.fit(X,y)

