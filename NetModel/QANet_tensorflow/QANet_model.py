import tensorflow as tf
from keras.models import load_model
import tensorflow_hub as hub
import keras.backend as K
from layers import regularizer, residual_block, highway, conv, mask_logits, optimized_trilinear_for_attention, \
    total_params


class Model(object):
    def __init__(self, config, word_mat=None, char_mat=None, test=False, use_elmo=False, use_cove=False):

        # hyper-parameter
        self.char_dim = config['char_dim']
        self.cont_limit = config['cont_limit'] if not test else 1000
        self.ques_limit = config['ques_limit'] if not test else 50
        self.char_limit = config['char_limit']
        self.ans_limit = config['ans_limit']
        self.filters = config['filters']
        self.num_heads = config['num_heads']
        self.batch_size = config['batch_size']
        self.l2_norm = config['l2_norm']
        self.decay = config['decay']
        self.learning_rate = config['learning_rate']
        self.grad_clip = config['grad_clip']
        self.use_elmo = use_elmo
        self.use_cove = use_cove
        self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")

        # embedding layer
        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(word_mat, dtype=tf.float32),
                                        trainable=False)
        self.char_mat = tf.get_variable("char_mat", initializer=tf.constant(char_mat, dtype=tf.float32), trainable=True)

        # input tensor
        self.contw_input_ = tf.placeholder(tf.int32, [None, self.cont_limit], "context_word")
        self.quesw_input_ = tf.placeholder(tf.int32, [None, self.ques_limit], "question_word")
        self.contc_input_ = tf.placeholder(tf.int32, [None, self.cont_limit, self.char_limit], "context_char")
        self.quesc_input_ = tf.placeholder(tf.int32, [None, self.ques_limit, self.char_limit], "question_char")
        self.y_start_ = tf.placeholder(tf.int32, [None, self.cont_limit + 1], "answer_start_index")
        self.y_end_ = tf.placeholder(tf.int32, [None, self.cont_limit + 1], "answer_end_index")
        self.contw_strings = tf.placeholder(tf.string, [None, self.cont_limit], 'contw_strings')
        self.quesw_strings = tf.placeholder(tf.string, [None, self.ques_limit], 'quesw_strings')

        self.c_mask = tf.cast(self.contw_input_, tf.bool)
        self.q_mask = tf.cast(self.quesw_input_, tf.bool)
        self.cont_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.ques_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)
        if self.use_elmo:
            elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
            self.cont_elmo = elmo(inputs={"tokens": self.contw_strings, "sequence_len": self.cont_len},
                                  signature="tokens", as_dict=True)["elmo"]
            self.ques_elmo = elmo(inputs={"tokens": self.quesw_strings, "sequence_len": self.ques_len},
                                  signature="tokens", as_dict=True)["elmo"]

        # if self.use_cove:
        #     self.cove_model = load_model('Keras_CoVe_V2.h5')
        #     self.cove_model.trainable = False

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
        if self.use_elmo:
            self.cont_elmo = tf.slice(self.cont_elmo, [0, 0, 0], [-1, self.c_maxlen, 1024])
            self.ques_elmo = tf.slice(self.ques_elmo, [0, 0, 0], [-1, self.q_maxlen, 1024])

        # init model & complie
        self.build_model()
        total_params()
        self.complie()

    def build_model(self):
        PL, QL, CL, d, dc, nh = self.c_maxlen, self.q_maxlen, self.char_limit, self.filters, self.char_dim, self.num_heads

        with tf.variable_scope("Input_Embedding_Layer"):
            ch_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.contc_input), [-1, CL, dc])
            qh_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.quesc_input), [-1, CL, dc])
            ch_emb = tf.nn.dropout(ch_emb, 1.0 - 0.5 * self.dropout)
            qh_emb = tf.nn.dropout(qh_emb, 1.0 - 0.5 * self.dropout)

            # Bidaf style conv-highway encoder
            ch_emb = conv(ch_emb, d, bias=True, activation=tf.nn.relu, kernel_size=5, name="char_conv", reuse=None)
            qh_emb = conv(qh_emb, d, bias=True, activation=tf.nn.relu, kernel_size=5, name="char_conv", reuse=True)

            ch_emb = tf.reduce_max(ch_emb, axis=1)
            qh_emb = tf.reduce_max(qh_emb, axis=1)

            ch_emb = tf.reshape(ch_emb, [-1, PL, ch_emb.shape[-1]])
            qh_emb = tf.reshape(qh_emb, [-1, QL, ch_emb.shape[-1]])

            c_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.contw_input), 1.0 - self.dropout)
            q_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.quesw_input), 1.0 - self.dropout)

            # if self.use_cove:
            #     c_emb_cove = self.cove_model(c_emb)
            #     q_emb_cove = self.cove_model(q_emb)
            #     c_emb = tf.concat([c_emb, c_emb_cove], axis=-1)
            #     q_emb = tf.concat([q_emb, q_emb_cove], axis=-1)

            c_emb = tf.concat([c_emb, ch_emb], axis=2)
            q_emb = tf.concat([q_emb, qh_emb], axis=2)

            if self.use_elmo:
                c_emb = tf.concat([c_emb, self.cont_elmo], axis=-1)
                q_emb = tf.concat([q_emb, self.ques_elmo], axis=-1)

            c_emb = highway(c_emb, size=d, scope="highway", dropout=self.dropout, reuse=None)
            q_emb = highway(q_emb, size=d, scope="highway", dropout=self.dropout, reuse=True)

        with tf.variable_scope("Embedding_Encoder_Layer"):
            c = residual_block(c_emb,
                               num_blocks=1,
                               num_conv_layers=4,
                               kernel_size=7,
                               mask=self.c_mask,
                               num_filters=d,
                               num_heads=nh,
                               seq_len=self.cont_len,
                               scope="Encoder_Residual_Block",
                               bias=False,
                               dropout=self.dropout)
            q = residual_block(q_emb,
                               num_blocks=1,
                               num_conv_layers=4,
                               kernel_size=7,
                               mask=self.q_mask,
                               num_filters=d,
                               num_heads=nh,
                               seq_len=self.ques_len,
                               scope="Encoder_Residual_Block",
                               reuse=True,
                               bias=False,
                               dropout=self.dropout)

        with tf.variable_scope("Context_to_Query_Attention_Layer"):
            S = optimized_trilinear_for_attention([c, q], self.c_maxlen, self.q_maxlen,
                                                  input_keep_prob=1.0 - self.dropout)
            mask_q = tf.expand_dims(self.q_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, mask=mask_q))
            mask_c = tf.expand_dims(self.c_mask, 2)
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask=mask_c), dim=1), (0, 2, 1))
            c2q = tf.matmul(S_, q)
            q2c = tf.matmul(tf.matmul(S_, S_T), c)
            attention_outputs = [c, c2q, c * c2q, c * q2c]

        with tf.variable_scope("Model_Encoder_Layer"):
            attention_inputs = tf.concat(attention_outputs, axis=-1)
            enc = [conv(attention_inputs, d, name="input_projection")]
            for i in range(3):
                if i % 2 == 0:  # dropout every 2 blocks
                    enc[i] = tf.nn.dropout(enc[i], 1.0 - self.dropout)
                enc.append(residual_block(enc[i],
                                          num_blocks=7,
                                          num_conv_layers=2,
                                          kernel_size=5,
                                          mask=self.c_mask,
                                          num_filters=d,
                                          num_heads=nh,
                                          seq_len=self.cont_len,
                                          scope="Model_Encoder",
                                          bias=False,
                                          reuse=True if i > 0 else None,
                                          dropout=self.dropout))

        with tf.variable_scope("Output_Layer"):
            start_logits = tf.concat([enc[1], enc[2]], axis=-1)
            end_logits = tf.concat([enc[1], enc[3]], axis=-1)
            if self.use_elmo:
                start_logits = tf.concat((start_logits, self.cont_elmo), axis=-1)
                end_logits = tf.concat((end_logits, self.cont_elmo), axis=-1)

            start_logits = tf.squeeze(conv(start_logits, 1, bias=False, name="start_pointer"), -1)
            end_logits = tf.squeeze(conv(end_logits, 1, bias=False, name="end_pointer"), -1)
            unanswer_bias = tf.get_variable("unanswer_bias", [1],
                                            regularizer=tf.contrib.layers.l2_regularizer(scale=3e-7),
                                            initializer=tf.zeros_initializer())
            unanswer_bias = tf.reshape(tf.tile(unanswer_bias, [self.batch_size]), [-1, 1])
            self.logits1 = tf.concat((unanswer_bias, mask_logits(start_logits, mask=self.c_mask)), axis=-1)
            self.logits2 = tf.concat((unanswer_bias, mask_logits(end_logits, mask=self.c_mask)), axis=-1)
            start_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits1, labels=self.y_start)
            end_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits2, labels=self.y_end)
            self.loss = tf.reduce_mean(start_loss + end_loss)
            if self.l2_norm is not None:
                variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                l2_loss = tf.contrib.layers.apply_regularization(regularizer, variables)
                self.loss += l2_loss

            # output
            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(self.logits1), axis=2),
                              tf.expand_dims(tf.nn.softmax(self.logits2), axis=1))
            outer = tf.matrix_band_part(outer, 0, self.ans_limit)
            self.output1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1) - 1
            self.output2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1) - 1

            if self.decay is not None:
                self.var_ema = tf.train.ExponentialMovingAverage(self.decay)
                ema_op = self.var_ema.apply(tf.trainable_variables())
                with tf.control_dependencies([ema_op]):
                    self.loss = tf.identity(self.loss)
                    self.assign_vars = []
                    for var in tf.global_variables():
                        v = self.var_ema.average(var)
                        if v is not None:
                            self.assign_vars.append(tf.assign(var, v))

    def complie(self):
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.lr = tf.minimum(self.learning_rate,
                             0.001 / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.8, beta2=0.999, epsilon=1e-7)
        grads = self.opt.compute_gradients(self.loss)
        gradients, variables = zip(*grads)
        capped_grads, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
        self.train_op = self.opt.apply_gradients(zip(capped_grads, variables), global_step=self.global_step)
