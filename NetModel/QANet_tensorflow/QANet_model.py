import tensorflow as tf
import tensorflow_hub as hub
import keras.backend as K
from layers import regularizer, residual_block, highway, conv, mask_logits, optimized_trilinear_for_attention, \
    total_params


class Model(object):
    def __init__(self, config, word_mat=None, char_mat=None, test=False, use_elmo=False):

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
        self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")
        self.use_topk = config['use_topk']
        self.diversity_loss = config['diversity_loss']
        self.topk_loss = config['topk_loss']

        # embedding layer
        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(word_mat, dtype=tf.float32),
                                        trainable=False)
        self.char_mat = tf.get_variable("char_mat", initializer=tf.constant(char_mat, dtype=tf.float32), trainable=True)

        # input tensor
        self.contw_input_ = tf.placeholder(tf.int32, [None, self.cont_limit], "context_word")
        self.quesw_input_ = tf.placeholder(tf.int32, [None, self.ques_limit], "question_word")
        self.contc_input_ = tf.placeholder(tf.int32, [None, self.cont_limit, self.char_limit], "context_char")
        self.quesc_input_ = tf.placeholder(tf.int32, [None, self.ques_limit, self.char_limit], "question_char")
        self.y_start_ = tf.placeholder(tf.int32, [None, self.cont_limit], "answer_start_index")
        self.y_end_ = tf.placeholder(tf.int32, [None, self.cont_limit], "answer_end_index")
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

        # slice for maxlen in each batch
        self.c_maxlen = tf.reduce_max(self.cont_len)
        self.q_maxlen = tf.reduce_max(self.ques_len)

        self.contw_input = tf.slice(self.contw_input_, [0, 0], [-1, self.c_maxlen])
        self.quesw_input = tf.slice(self.quesw_input_, [0, 0], [-1, self.q_maxlen])
        self.c_mask = tf.slice(self.c_mask, [0, 0], [-1, self.c_maxlen])
        self.q_mask = tf.slice(self.q_mask, [0, 0], [-1, self.q_maxlen])
        self.contc_input = tf.slice(self.contc_input_, [0, 0, 0], [-1, self.c_maxlen, self.char_limit])
        self.quesc_input = tf.slice(self.quesc_input_, [0, 0, 0], [-1, self.q_maxlen, self.char_limit])
        self.y_start = tf.slice(self.y_start_, [0, 0], [-1, self.c_maxlen])
        self.y_end = tf.slice(self.y_end_, [0, 0], [-1, self.c_maxlen])
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
            # 2.0 Dataset
            # unanswer_bias = tf.get_variable("unanswer_bias", [1],
            #                                 regularizer=tf.contrib.layers.l2_regularizer(scale=3e-7),
            #                                 initializer=tf.zeros_initializer())
            # unanswer_bias = tf.reshape(tf.tile(unanswer_bias, [self.batch_size]), [-1, 1])
            # self.logits1 = tf.concat((unanswer_bias, mask_logits(start_logits, mask=self.c_mask)), axis=-1)
            # self.logits2 = tf.concat((unanswer_bias, mask_logits(end_logits, mask=self.c_mask)), axis=-1)
            self.logits1 = mask_logits(start_logits, mask=self.c_mask)
            self.logits2 = mask_logits(end_logits, mask=self.c_mask)
            start_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits1, labels=self.y_start)
            end_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits2, labels=self.y_end)
            self.loss = tf.reduce_mean(start_loss + end_loss)

            # output
            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(self.logits1), axis=2),
                              tf.expand_dims(tf.nn.softmax(self.logits2), axis=1))
            outer = tf.matrix_band_part(outer, 0, self.ans_limit)
            self.output1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            self.output2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)

        if self.use_topk:
            with tf.variable_scope("Topk_Layer"):
                top_size = 3
                outer = tf.reshape(outer, [self.batch_size, -1])
                outer_inds = tf.nn.top_k(outer, top_size).indices  # [N,top_size]
                self.yp1 = outer_inds // tf.shape(self.logits1)[-1]
                self.yp2 = outer_inds % tf.shape(self.logits2)[-1]

                def sen_mask(tensor):
                    def sen_mask_(a, b, filters):
                        try:
                            mata = tf.zeros([a, filters], tf.int32)
                        except:
                            mata = []
                        matb = tf.ones([b - a, filters], tf.int32)
                        matc = tf.zeros([tf.shape(self.logits1)[-1] - b, filters], tf.int32)
                        mat = tf.concat((mata, matb, matc), axis=0)
                        return mat

                    return tf.map_fn(lambda x: sen_mask_(x[0], x[1], self.filters), tensor)

                self.yp3 = self.yp2 + 1
                self.yp1 = tf.expand_dims(self.yp1, -1)
                self.yp2 = tf.expand_dims(self.yp2, -1)
                self.yp3 = tf.expand_dims(self.yp3, -1)
                self.y_mask = tf.concat([self.yp1, self.yp3], axis=-1)
                self.y_mask = tf.map_fn(lambda x: sen_mask(x), self.y_mask)

                # answer
                c = tf.tile(tf.expand_dims(c2q, 1), [1, top_size, 1, 1])
                c_topk = tf.multiply(tf.cast(self.y_mask, tf.float32), c)
                W1 = tf.get_variable("W1", initializer=tf.ones([1, 1, 1, self.filters]))
                W1 = tf.tile(W1, [self.batch_size, top_size, 1, 1])
                alpha1 = tf.nn.softmax(tf.matmul(W1, c_topk, transpose_b=True), axis=2)
                answer = tf.matmul(alpha1, c_topk)  # [32,top_size,1,128]

                # question
                W2 = tf.get_variable("W2", initializer=tf.ones([1, 1, self.filters]))
                W2 = tf.tile(W2, [self.batch_size, 1, 1])
                alpha2 = tf.nn.softmax(tf.matmul(W2, q, transpose_b=True), axis=1)
                ques = tf.matmul(alpha2, q)
                ques = tf.tile(tf.expand_dims(ques, 1), [1, top_size, 1, 1])  # [32,top_size,1,128]

                # question & answer
                W3 = tf.get_variable("W3", initializer=tf.ones([1, 1, self.filters, self.filters]))
                W3 = tf.tile(W3, [self.batch_size, top_size, 1, 1])
                y_topk_logits = tf.nn.sigmoid(tf.matmul(ques, tf.matmul(W3, answer, transpose_b=True))) # [32,top_size,1,1]
                y_topk_logits = tf.squeeze(y_topk_logits)  # [32,top_size]

                self.yp1 = tf.squeeze(self.yp1)
                self.yp2 = tf.squeeze(self.yp2)
                coeff1_topk = tf.one_hot(self.yp1, self.c_maxlen, axis=-1) # [32,top_size,400] one-hot
                coeff2_topk = tf.one_hot(self.yp2, self.c_maxlen, axis=-1)
                # [0,1,0,0,0][0,0,0,1,0]->[0,1,1,1,1]-[0,0,0,1,1]->[0,1,1,0,0]+[0,0,0,1,0]->[0,1,1,1,0]
                coeff1_topk_cumsum = tf.cumsum(coeff1_topk, axis=-1)
                coeff2_topk_cumsum = tf.cumsum(coeff2_topk, axis=-1)
                self.y_d = coeff1_topk_cumsum - coeff2_topk_cumsum + coeff2_topk # [32, top_size, 400]

                def clip_for_sigmoid(output):
                    _epsilon = tf.convert_to_tensor(1e-7, dtype=output.dtype.base_dtype)
                    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
                    output = tf.log(output / (1 - output))
                    return output

                if self.topk_loss=='f1':
                    # f1 loss
                    y_start_ind = tf.cumsum(self.y_start, axis=-1)
                    y_end_ind = tf.cumsum(self.y_end, axis=-1)
                    y_gtd = y_start_ind - y_end_ind + self.y_end # [32, 400]
                    def cal_num_same(y_pred, y_truth): # [top_size, 400] [400,]
                        def cal_num_same_(y_pred_, y_truth): # [400,] [400,]
                            return tf.reduce_sum(tf.cast(tf.logical_and(tf.cast(y_pred_, tf.bool), tf.cast(y_truth, tf.bool)), tf.float32),axis=-1)
                        return [tf.map_fn(lambda x:cal_num_same_(x,y_truth),y_pred),tf.map_fn(lambda x:cal_num_same_(x,y_truth),y_pred)]
                    num_same = tf.map_fn(lambda x:cal_num_same(x[0], x[1]), [self.y_d, y_gtd])[0] # [32, top_size]
                    y_precision = num_same / (tf.cast(tf.reduce_sum(self.y_d, axis=-1),tf.float32) + 1e-8) # [32, top_size]
                    y_recall = num_same / tf.expand_dims(tf.cast(tf.reduce_sum(y_gtd, axis=-1),tf.float32) + 1e-8, axis=-1) # [32, top_size]
                    y_f1 = (2.0 * y_precision * y_recall) / (tf.cast(y_precision + y_recall,tf.float32) + 1e-8) # [32, top_size]
                    topk_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=clip_for_sigmoid(y_topk_logits), labels=y_f1))

                elif self.topk_loss=='em':
                    # em loss
                    start_em = tf.equal(tf.cast(tf.expand_dims(tf.argmax(self.y_start, axis=-1), axis=1), tf.int32),
                                        tf.cast(self.yp1, tf.int32))  # [32, top_size]
                    end_em = tf.equal(tf.cast(tf.expand_dims(tf.argmax(self.y_end, axis=-1), axis=1), tf.int32),
                                      tf.cast(self.yp2, tf.int32))  # [32, top_size]
                    y_em = tf.cast(tf.logical_and(start_em, end_em), tf.float32) # [32, top_size]
                    topk_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=clip_for_sigmoid(y_topk_logits), labels=y_em))

                # final loss
                self.Lambda1 = tf.get_variable("Lambda1", initializer=tf.constant([0.9]), trainable=False)
                self.loss = tf.reduce_mean(self.Lambda1 * (start_loss + end_loss) + (1 - self.Lambda1) * topk_loss)

                # output
                outer_topk = tf.matmul(tf.expand_dims(tf.nn.softmax(self.logits1), axis=2),
                                  tf.expand_dims(tf.nn.softmax(self.logits2), axis=1))
                outer_topk = tf.matrix_band_part(outer_topk, 0, self.ans_limit)
                self.output1 = tf.argmax(tf.reduce_max(outer_topk, axis=2), axis=1)
                self.output2 = tf.argmax(tf.reduce_max(outer_topk, axis=1), axis=1)

                # diversity loss
                if self.diversity_loss:
                    self.Lambda2 = tf.get_variable("Lambda2", initializer=tf.constant([0.1]),trainable=False)
                    diversity_loss = tf.reduce_mean(tf.reduce_prod(self.y_d, axis=1),axis=-1) # [32,top_size,400]->[32,400]->[32,]
                    self.loss = self.loss + tf.reduce_mean(self.Lambda2 * diversity_loss)


        if self.l2_norm is not None:
            variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            l2_loss = tf.contrib.layers.apply_regularization(regularizer, variables)
            self.loss += l2_loss

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
