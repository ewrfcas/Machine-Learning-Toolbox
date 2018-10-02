import tensorflow as tf
from keras.models import load_model
import math
from tensorflow.contrib.layers import apply_regularization
from NetModel.SAN.SAN_layers import total_params, CharCNN, FeedForward, BiLSTM, DeepAttentionLayers, AttentionLayer, SumAttention, SAN, Dense, regularizer

class Model(object):
    def __init__(self, config, word_mat=None, char_mat=None, test=False):
        # hyper-parameter
        self.char_dim = config['char_dim']
        self.cont_limit = config['cont_limit'] if not test else 1000
        self.ques_limit = config['ques_limit'] if not test else 50
        self.char_limit = config['char_limit']
        self.ans_limit = config['ans_limit']
        self.filters = config['filters']
        self.batch_size = config['batch_size']
        self.l2_norm = config['l2_norm']
        self.gamma = config['gamma']
        self.decay = config['decay']
        self.learning_rate = config['learning_rate']
        self.grad_clip = config['grad_clip']
        self.optimizer = config['optimizer']
        self.use_elmo = config['use_elmo']
        self.use_cove = config['use_cove']
        self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")
        self.dropout_rnn = tf.placeholder_with_default(0.0, (), name="dropout_rnn")
        self.dropout_emb = tf.placeholder_with_default(0.0, (), name="dropout_emb")
        self.dropout_att = tf.placeholder_with_default(0.0, (), name="dropout_att")
        self.un_size = tf.placeholder_with_default(self.batch_size, (), name="un_size")

        # input tensor
        self.contw_input_ = tf.placeholder(tf.int32, [None, self.cont_limit], "context_word")
        self.quesw_input_ = tf.placeholder(tf.int32, [None, self.ques_limit], "question_word")
        self.contc_input_ = tf.placeholder(tf.int32, [None, self.cont_limit, self.char_limit], "context_char")
        self.quesc_input_ = tf.placeholder(tf.int32, [None, self.ques_limit, self.char_limit], "question_char")
        self.y_start_ = tf.placeholder(tf.int32, [None, self.cont_limit + 1], "answer_start_index")
        self.y_end_ = tf.placeholder(tf.int32, [None, self.cont_limit + 1], "answer_end_index")

        self.c_mask = tf.cast(self.contw_input_, tf.bool)
        self.q_mask = tf.cast(self.quesw_input_, tf.bool)
        self.cont_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.ques_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

        # embedding layer
        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(word_mat, dtype=tf.float32), trainable=False)
        with tf.variable_scope("Input_Embedding_Mat"):
            self.char_mat = tf.get_variable("char_mat", initializer=tf.constant(char_mat, dtype=tf.float32), trainable=True)

        # slice for maxlen in each batch
        self.c_maxlen = tf.reduce_max(self.cont_len)
        self.q_maxlen = tf.reduce_max(self.ques_len)

        self.contw_input = tf.slice(self.contw_input_, [0, 0], [-1, self.c_maxlen])
        self.quesw_input = tf.slice(self.quesw_input_, [0, 0], [-1, self.q_maxlen])
        self.c_mask = tf.slice(self.c_mask, [0, 0], [-1, self.c_maxlen])
        self.q_mask = tf.slice(self.q_mask, [0, 0], [-1, self.q_maxlen])
        self.contc_input = tf.slice(self.contc_input_, [0, 0, 0], [-1, self.c_maxlen, self.char_limit])
        self.quesc_input = tf.slice(self.quesc_input_, [0, 0, 0], [-1, self.q_maxlen, self.char_limit])
        self.y_start = tf.slice(self.y_start_, [0, 0], [-1, self.c_maxlen + 1])
        self.y_end = tf.slice(self.y_end_, [0, 0], [-1, self.c_maxlen + 1])

        if self.use_cove == 2:
            with tf.variable_scope('Cove_Layer'):
                self.cove_model = load_model(config['cove_path'])
        else:
            self.cove_cont_low_ = tf.placeholder(tf.float32, [None, self.cont_limit, 600], 'cove_cont_low')
            self.cove_cont_high_ = tf.placeholder(tf.float32, [None, self.cont_limit, 600], 'cove_cont_high')
            self.cove_ques_low_ = tf.placeholder(tf.float32, [None, self.ques_limit, 600], 'cove_ques_low')
            self.cove_ques_high_ = tf.placeholder(tf.float32, [None, self.ques_limit, 600], 'cove_ques_high')
            self.cove_cont_low = tf.slice(self.cove_cont_low_, [0, 0, 0], [-1, self.c_maxlen, 600])
            self.cove_cont_high = tf.slice(self.cove_cont_high_, [0, 0, 0], [-1, self.c_maxlen, 600])
            self.cove_ques_low = tf.slice(self.cove_ques_low_, [0, 0, 0], [-1, self.q_maxlen, 600])
            self.cove_ques_high = tf.slice(self.cove_ques_high_, [0, 0, 0], [-1, self.q_maxlen, 600])

        # lr schedule
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)

        self.learning_rate = tf.placeholder_with_default(config['learning_rate'], (), name="learning_rate")
        self.lr = tf.minimum(self.learning_rate,
                             self.learning_rate / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))

        # init model & complie
        self.build_model()
        total_params(exclude='Cove_Layer')
        self.complie()

    def build_model(self):
        with tf.variable_scope("Input_Embedding_Layer"):
            with tf.variable_scope('Char_Conv', reuse=tf.AUTO_REUSE):
                ch_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.char_mat, self.contc_input), 1.0 - self.dropout_emb)
                qh_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.char_mat, self.quesc_input), 1.0 - self.dropout_emb)
                ch_emb = CharCNN(ch_emb, self.char_limit, self.char_dim, self.filters, self.c_maxlen)
                qh_emb = CharCNN(qh_emb, self.char_limit, self.char_dim, self.filters, self.q_maxlen)

            c_emb0 = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.contw_input), 1.0 - self.dropout_emb)
            q_emb0 = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.quesw_input), 1.0 - self.dropout_emb)

            # cove features
            if self.use_cove == 2:
                cove_cont_low, cove_cont_high = tf.unstack(tf.stop_gradient(self.cove_model(c_emb0)), axis=0)
                cove_ques_low, cove_ques_high = tf.unstack(tf.stop_gradient(self.cove_model(q_emb0)), axis=0)
            else:
                cove_cont_low, cove_cont_high = self.cove_cont_low, self.cove_cont_high
                cove_ques_low, cove_ques_high = self.cove_ques_low, self.cove_ques_high

            # pre alignment
            c2q_prealign = AttentionLayer(c_emb0, q_emb0, q_emb0, self.q_mask, self.filters, self.dropout_att)

            c_emb = tf.concat([c_emb0, ch_emb, cove_cont_low], axis=-1)
            q_emb = tf.concat([q_emb0, qh_emb, cove_ques_low], axis=-1)
            c_emb = tf.nn.dropout(c_emb, 1 - self.dropout)
            q_emb = tf.nn.dropout(q_emb, 1 - self.dropout)

            # FeedForward layer
            with tf.variable_scope('FeedForward_Layer'):
                c_emb = FeedForward(c_emb, self.filters, self.dropout, name='cont_ff')
                q_emb = FeedForward(q_emb, self.filters, self.dropout, name='ques_ff')

        with tf.variable_scope('Encoder_Layers'):
            with tf.variable_scope('Contextual_Encoder', reuse=tf.AUTO_REUSE):
                # context encode
                c_emb_low = tf.concat([c_emb, c2q_prealign, cove_cont_low], axis=-1)
                c_emb_low = BiLSTM(c_emb_low, filters=self.filters, name='cont_lstm_low', dropout=self.dropout_rnn)
                c_emb_high = tf.concat([c_emb_low, cove_cont_high], axis=-1)
                c_emb_high = BiLSTM(c_emb_high, filters=self.filters, name='cont_lstm_high', dropout=self.dropout_rnn)
                c_emb_high = tf.nn.dropout(c_emb_high, 1 - self.dropout)
                # question encode
                q_emb_low = tf.concat([q_emb, cove_ques_low], axis=-1)
                q_emb_low = BiLSTM(q_emb_low, filters=self.filters, name='ques_lstm_low', dropout=self.dropout_rnn)
                q_emb_high = tf.concat([q_emb_low, cove_ques_high], axis=-1)
                q_emb_high = BiLSTM(q_emb_high, filters=self.filters, name='ques_lstm_high', dropout=self.dropout_rnn)
                q_mem_hidden = BiLSTM(tf.concat([q_emb_low, q_emb_high], axis=-1), self.filters, self.dropout_rnn, name='ques_lstm_memory')

            # c2q encode
            with tf.variable_scope('C2Q_Attention_Encoder'):
                c_att_input = tf.concat([c_emb0, cove_cont_high, c_emb_low, c_emb_high], axis=-1)
                q_att_input = tf.concat([q_emb0, cove_ques_high, q_emb_low, q_emb_high], axis=-1)
                v_att_input = [q_emb_low, q_emb_high, q_mem_hidden]
                c2q_att_hidden = DeepAttentionLayers(c_att_input, q_att_input, v_att_input, self.q_mask, self.filters,
                                                     self.dropout_att, name='C2Q_Attention')
                c_mem_hidden = BiLSTM(tf.concat([c2q_att_hidden, c_emb_low, c_emb_high], axis=-1),
                                      self.filters, self.dropout_rnn, name='cont_lstm_memory')
            # self attention
            with tf.variable_scope('Self_Attention_Encoder'):
                c_mem_input = tf.concat([c2q_att_hidden, c_mem_hidden, c_emb_low,
                                         c_emb_high, cove_cont_high, c_emb0], axis=-1)
                c_self_hidden = AttentionLayer(c_mem_input, c_mem_input, c_mem_input,
                                               self.c_mask, self.filters, self.dropout_att)
                c_mem = BiLSTM(tf.concat([c_self_hidden, c_mem_hidden], axis=-1),
                               self.filters, self.dropout_rnn, name='cont_self_memory')
                q_mem = SumAttention(q_mem_hidden, self.q_mask, self.dropout_att)

        with tf.variable_scope('Point_Network'):
            start_scores, end_scores = SAN(c_mem, q_mem, self.c_mask, filters=self.filters*2,
                                           hidden_size=self.filters*2, num_turn=5, name='SAN', dropout=self.dropout)
            self.unanswer_bias = tf.get_variable("unanswer_bias", [1], initializer=tf.zeros_initializer())
            self.unanswer_bias = tf.reshape(tf.tile(self.unanswer_bias, [self.un_size]), [-1, 1])
            start_scores = tf.concat((self.unanswer_bias, start_scores), axis=-1)
            end_scores = tf.concat((self.unanswer_bias, end_scores), axis=-1)
            c_sum = SumAttention(c_mem, self.c_mask, self.dropout_att)
            pred_score = Dense(tf.concat([c_sum, q_mem], axis=-1), 1, norm=True, dropout=self.dropout)

        with tf.variable_scope('Loss_Layer'):
            start_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=start_scores, labels=self.y_start)
            end_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=end_scores, labels=self.y_end)
            self.loss_a = tf.reduce_mean(start_loss + end_loss)
            answer_exist_label = tf.squeeze(tf.cast(tf.slice(self.y_start, [0, 0], [-1, 1]), tf.float32), axis=-1)
            self.loss_c = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_score, labels=answer_exist_label))
            self.loss = self.loss_a + self.gamma * self.loss_c
            # l2 loss
            if self.l2_norm is not None:
                variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                l2_loss = apply_regularization(regularizer, variables)
                self.loss += self.l2_norm * tf.reduce_sum(l2_loss)

        with tf.variable_scope('Output_Layer'):
            unanswer_mask = tf.cast(tf.less(tf.nn.sigmoid(pred_score), 0.5), tf.int64)  # [bs,] has answer=1 no answer=0
            unanswer_move = unanswer_mask - 1  # [bs,] has answer=0 no answer=-1
            softmax_start_scores = tf.nn.softmax(tf.slice(start_scores, [0, 1], [-1, -1]))
            softmax_end_scores = tf.nn.softmax(tf.slice(end_scores, [0, 1], [-1, -1]))
            outer = tf.matmul(tf.expand_dims(softmax_start_scores, axis=2),
                              tf.expand_dims(softmax_end_scores, axis=1))
            outer = tf.matrix_band_part(outer, 0, self.ans_limit)
            def position_encoding(x):
                for i in range(x.shape[0]):
                    for j in range(x.shape[1]):
                        if j - i > 5:
                            x[i][j] = float(1.0 / math.log(j - i + 1))
                return x
            mask_mat = tf.ones((self.c_maxlen, self.c_maxlen))
            mask_mat = tf.expand_dims(tf.py_func(position_encoding, [mask_mat], tf.float32), axis=0)
            mask_mat = tf.tile(mask_mat, [self.un_size, 1, 1])

            outer_masked = outer * mask_mat
            self.mask_output1 = tf.argmax(tf.reduce_max(outer_masked, axis=2), axis=1) * unanswer_mask + unanswer_move
            self.mask_output2 = tf.argmax(tf.reduce_max(outer_masked, axis=1), axis=1) * unanswer_mask + unanswer_move

    def complie(self):
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.8, beta2=0.999, epsilon=1e-7)
        grads = self.opt.compute_gradients(self.loss)
        gradients, variables = zip(*grads)
        capped_grads, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
        train_op = self.opt.apply_gradients(zip(capped_grads, variables), global_step=self.global_step)

        # EMA
        with tf.variable_scope("EMA_Weights"):
            if self.decay is not None:
                self.var_ema = tf.train.ExponentialMovingAverage(self.decay)
                with tf.control_dependencies([train_op]):
                    self.ema_train_op = self.var_ema.apply(
                        list(set(tf.trainable_variables()) ^ set(tf.trainable_variables('Cove_Layer'))))
                # assign ema weights
                self.assign_vars = []
                for var in tf.global_variables():
                    v = self.var_ema.average(var)
                    if v is not None:
                        self.assign_vars.append(tf.assign(var, v))

import numpy as np
config = {
    'char_dim': 64,
    'cont_limit': 400,
    'ques_limit': 50,
    'char_limit': 16,
    'ans_limit': -1,
    'filters': 256,
    'dropout': 0.1,
    'dropout_emb': 0.1,
    'l2_norm': 3e-7,
    'decay': 0.9999,
    'gamma': 1.0,
    'learning_rate': 1e-3,
    'grad_clip': 5.0,
    'use_elmo': 0,
    'use_cove': 0,
    'optimizer': 'adam',
    'cove_path': 'Keras_CoVe_2layers.h5',
    'train_tfrecords': 'tfrecords/train_pre_elmo_cove.tfrecords',
    'dev_tfrecords': 'tfrecords/dev_pre_elmo_cove.tfrecords',
    'batch_size': 24,
    'epoch': 40,
    'origin_path': None,  # not finetune
    'path': 'QANetV253'
}
word_mat = np.random.random((90950, 300))
char_mat = np.random.random((1223, 64))
model = Model(config, word_mat, char_mat)