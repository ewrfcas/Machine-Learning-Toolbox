# ! -*- coding: utf-8 -*-
from keras.engine.topology import Layer
from keras.regularizers import *
import tensorflow as tf
import keras.backend as K

class GateAttention(Layer):
    def __init__(self, filters, dropout=0.0, regularizer=l2(3e-7), **kwargs):
        self.filters = filters
        self.dropout = dropout
        self.regularizer = regularizer
        super(GateAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WC = self.add_weight(name='WC',
                                  shape=(input_shape[0][-1], self.filters),
                                  regularizer=self.regularizer,
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[1][-1], self.filters),
                                  regularizer=self.regularizer,
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.V = self.add_weight(name='V',
                                 shape=(2 * input_shape[1][-1], self.filters),
                                 regularizer=self.regularizer,
                                 initializer='glorot_uniform',
                                 trainable=True)
        super(GateAttention, self).build(input_shape)

    def mask_logits(self, inputs, mask, clen, mask_value=-1e12):
        shapes = [x if x != None else -1 for x in inputs.shape.as_list()]
        mask = K.cast(mask, tf.int32)
        mask = K.one_hot(mask[:, 0], shapes[-1])
        mask = 1 - K.cumsum(mask, 1)
        mask = tf.cast(mask, tf.float32)
        mask = tf.tile(tf.expand_dims(mask, axis=1), [1, clen, 1])
        return inputs + mask_value * (1 - mask)

    def call(self, x, mask=None):
        x_cont, x_ques, ques_len = x
        input_shape_ = x_cont.shape.as_list()
        x_cont_ = tf.nn.relu(K.dot(x_cont, self.WC))
        x_ques_ = tf.nn.relu(K.dot(x_ques, self.WQ))
        logits = tf.matmul(x_cont_, x_ques_, transpose_b=True) / (self.filters ** 0.5)
        logits = self.mask_logits(logits, ques_len, clen=input_shape_[1])
        logits = tf.nn.softmax(logits)
        C = tf.matmul(logits, x_ques)
        res = tf.concat([x_cont, C], axis=2)
        gate = tf.nn.sigmoid(K.dot(res, self.V))
        return gate

    def compute_output_shape(self, input_shape):
        return input_shape[0]