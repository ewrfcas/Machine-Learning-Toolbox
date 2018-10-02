# stochastic answer network in tensorflow and tensor2tensor
import tensorflow as tf
from tensorflow.contrib.keras import layers
from tensorflow.contrib.layers import variance_scaling_initializer, l2_regularizer
from tensor2tensor.layers.common_layers import conv1d, dense, layer_norm

initializer = lambda: variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True, dtype=tf.float32)
initializer_relu = lambda: variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, dtype=tf.float32)
regularizer = l2_regularizer(scale=3e-7)

def exp_mask(inputs, mask, mask_value=-1e30):
    mask = tf.cast(mask, tf.float32)
    return inputs + mask_value * (1 - mask)

def GRUCell(x, h, hidden_size, filters):
    with tf.variable_scope('Output_GRU_Cell', reuse=tf.AUTO_REUSE):
        gate_kernel = tf.get_variable("gate_kernel",
                                      shape=(hidden_size + filters, 2 * hidden_size),
                                      dtype=tf.float32,
                                      initializer=initializer(),
                                      regularizer=regularizer)
        gate_bias = tf.get_variable("gates_bias",
                                    shape=(2 * hidden_size),
                                    dtype=tf.float32,
                                    regularizer=regularizer)
        candidate_kernel = tf.get_variable("candidate_kernel",
                                           shape=(hidden_size + filters, hidden_size),
                                           dtype=tf.float32,
                                           initializer=initializer(),
                                           regularizer=regularizer)
        candidate_bias = tf.get_variable("candidate_bias",
                                         shape=(hidden_size),
                                         dtype=tf.float32,
                                         regularizer=regularizer)

        gate_inputs = tf.matmul(tf.concat([x, h], axis=-1), gate_kernel) + gate_bias # [bs, f+h]*[bs, f+h, 2h]
        value = tf.nn.sigmoid(gate_inputs)
        r, u = tf.split(value=value, num_or_size_splits=2, axis=1) # [bs, h]*2
        r_state = r * h # [bs, h]
        candidate = tf.matmul(tf.concat([x, r_state], axis=-1), candidate_kernel) + candidate_bias # [bs, f+h]*[bs,f+h, h]
        c = tf.nn.tanh(candidate)
        new_h = u * h + (1 - u) * c # [bs, h]
    return new_h

def BilinearAttention(c, q, c_mask, filters, name, norm=False):
    q = tf.expand_dims(dense(q, filters, name=name, reuse=tf.AUTO_REUSE,
                             kernel_initializer=initializer(), kernel_regularizer=regularizer),axis=-1)  # [bs, dim, 1]
    if norm:
        q = layer_norm(q)
    cq = tf.squeeze(tf.matmul(c, q), axis=-1) # [bs, c_len, dim] * [bs, dim, 1] -> [bs, c_len]
    cq = exp_mask(cq, c_mask)
    return cq

def SAN(c_mem, q_mem, c_mask, filters, hidden_size, num_turn, name, dropout, type='last'):
    with tf.variable_scope(name):
        start_scores_list = []
        end_scores_list = []
        for turn in range(num_turn):
            st_scores = BilinearAttention(c_mem, q_mem, c_mask, filters, name='start_output') # [bs, c_len]
            start_scores_list.append(st_scores)
            end_scores = BilinearAttention(c_mem, q_mem, c_mask, filters, name='end_output') # [bs, c_len]
            end_scores_list.append(end_scores)
            x = tf.squeeze(tf.matmul(tf.expand_dims(tf.nn.softmax(end_scores), axis=1), c_mem), axis=1) # [bs, 1, c_len] * [bs, c_len, f]->[bs, 1, f]->[bs, f]
            q_mem = tf.nn.dropout(q_mem, 1.0 - dropout) # [bs, h]
            q_mem = GRUCell(x, q_mem, hidden_size=hidden_size, filters=filters) # [bs, h]

    if type=='last':
        return start_scores_list[-1], end_scores_list[-1]
    else:
        return start_scores_list, end_scores_list

def CharCNN(x, char_limit, char_dim, filters, maxlen, kernel_size=5, name='char_conv'):
    x = tf.reshape(x, [-1, char_limit, char_dim])
    x = tf.nn.relu(conv1d(x, filters, kernel_size=kernel_size, name=name, padding='same',
                          kernel_initializer=initializer_relu(), kernel_regularizer=regularizer))
    x = tf.reduce_max(x, axis=1)
    x = tf.reshape(x, [-1, maxlen, filters])
    return x

def FeedForward(x, filters, dropout, name):
    with tf.variable_scope(name):
        x = tf.nn.relu(conv1d(x, filters, kernel_size=1, padding='same', name='FFN_1',
                              kernel_initializer=initializer_relu(), kernel_regularizer=regularizer))
        x = conv1d(x, filters, kernel_size=1, padding='same', name = "FFN_2",
                   kernel_initializer=initializer(), kernel_regularizer=regularizer)
        x = tf.nn.dropout(x, 1 - dropout)
    return x

# keras based
def BiLSTM_keras(x=None, filters=256, dropout=0.0, name='BiLSTM', return_sequences=True):
    BiLSTM_Layer = layers.Bidirectional(layers.LSTM(filters, return_sequences=return_sequences),
                                        merge_mode='concat', name=name)
    if x is None:
        return BiLSTM_Layer
    else:
        x = BiLSTM_Layer(x)
        x = tf.nn.dropout(x, 1 - dropout)
        return x

# danamicLstm
# from tensorflow.contrib.rnn import LSTMCell
# from tensorflow.contrib.rnn import MultiRNNCell
# def BiLSTM(x, x_length, filters, dropout=0.0, name='BiLSTM'):
#     lstm_fw_cell = MultiRNNCell([LSTMCell(filters)])
#     lstm_bw_cell = MultiRNNCell([LSTMCell(filters)])
#     outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x,
#                                                   sequence_length=x_length, dtype=tf.float32, scope=name)
#     outputs = tf.concat(outputs, axis=-1)
#     outputs = tf.nn.dropout(outputs, 1 - dropout)
#     return outputs

# cudnnLSTM
from tensorflow.contrib.cudnn_rnn import CudnnLSTM
def BiLSTM(x, filters, dropout=0.0, name='BiLSTM'):
    cudnn_lstm = CudnnLSTM(1, filters, direction='bidirectional', name=name)
    x, _ = cudnn_lstm(x)
    x = tf.nn.dropout(x, 1 - dropout)
    return x

def Dense(x, unit, norm=True, dropout=0.0):
    x = dense(x, unit, kernel_initializer=initializer(), kernel_regularizer=regularizer)
    if norm:
        x = layer_norm(x)
    x = tf.nn.dropout(x, 1 - dropout)
    if unit==1:
        x = tf.squeeze(x, axis=-1)
    return x

def DotProductProject(x1, x2, filters, dropout):
    x1 = tf.nn.dropout(x1, 1 - dropout) # [bs, c_len, dim]
    x2 = tf.nn.dropout(x2, 1 - dropout) # [bs, q_len, dim]
    x1 = conv1d(x1, filters, kernel_size=1, padding='same', name='conv', reuse=tf.AUTO_REUSE,
                kernel_initializer=initializer_relu(), kernel_regularizer=regularizer) # [bs, c_len, filters]
    x1 = tf.nn.relu(layer_norm(x1))
    x2 = conv1d(x2, filters, kernel_size=1, padding='same', name='conv', reuse=tf.AUTO_REUSE,
                kernel_initializer=initializer_relu(), kernel_regularizer=regularizer) # [bs, q_len, filters]
    x2 = tf.nn.relu(layer_norm(x2))
    S = tf.matmul(x1, x2, transpose_b=True) # [bs, c_len, q_len]
    return S

def SimilarityMatrix(x1, x2, filters, type='dot_product_project', dropout=0.0):
    if type=='dot_product_project':
        S = DotProductProject(x1, x2, filters, dropout)
    else:
        raise NotImplementedError
    return S

def AttentionLayer(c, q, v, q_mask, filters, dropout):
    S = SimilarityMatrix(c, q, filters, dropout=dropout)  # [bs, c_len, q_len]
    q_mask = tf.expand_dims(q_mask, axis=1)  # [bs, q_len] -> [bs, 1, q_len]
    S = tf.nn.softmax(exp_mask(S, q_mask))
    S = tf.nn.dropout(S, 1 - dropout)
    x = tf.matmul(S, v)
    return x

def DeepAttentionLayers(c, q, v, q_mask, filters, dropout, name='DeepAttentionLayers'):
    att_outputs = []
    for i in range(len(v)):
        with tf.variable_scope(name+'_layer'+str(i+1)):
            x = AttentionLayer(c, q, v[i], q_mask, filters, dropout)
            att_outputs.append(x)
    att_outputs = tf.concat(att_outputs, axis=-1)
    att_outputs = tf.nn.dropout(att_outputs, 1 - dropout)
    return att_outputs

def SumAttention(x, mask, dropout):
    x = tf.nn.dropout(x, 1 - dropout)
    alpha = tf.squeeze(conv1d(x, 1, 1, kernel_initializer=initializer(), name='sum_conv',
                              kernel_regularizer=regularizer), axis=-1)  # [bs, c_len]
    alpha = exp_mask(alpha, mask) # [bs, c_len]
    alpha = tf.expand_dims(tf.nn.softmax(alpha), axis=1) # [bs, 1, c_len]
    x = tf.squeeze(tf.matmul(alpha, x), axis=1) # x:[bs, c_len, dim] -> [bs, dim]
    return x

def total_params(exclude=None):
    total_parameters = 0
    if exclude is not None:
        trainable_variables = list(set(tf.trainable_variables())^set(tf.trainable_variables(exclude)))
    else:
        trainable_variables = tf.trainable_variables()
    for variable in trainable_variables:
        shape = variable.get_shape()
        variable_parametes = 1
        try:
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        except:
            print(shape,'cudnn weights is unknown')
    print("Total number of trainable parameters: {}".format(total_parameters))