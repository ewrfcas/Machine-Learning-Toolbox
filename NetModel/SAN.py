# stochastic answer network in tensorflow and tensor2tensor
import tensorflow as tf
from tensor2tensor.layers.common_layers import conv1d, dense

regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7)
initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True, dtype=tf.float32)

def exp_mask(inputs, mask, mask_value=-1e30):
    mask = tf.cast(mask, tf.float32)
    return inputs + mask_value * (1 - mask)

def linear_sum_attention(x, mask, dropout):
    alpha = tf.squeeze(conv1d(x, 1, 1, kernel_initializer=initializer, kernel_regularizer=regularizer), axis=-1)  # [bs, c_len]
    alpha = exp_mask(alpha, mask) # [bs, c_len]
    alpha = tf.expand_dims(tf.nn.softmax(alpha), axis=1) # [bs, 1, c_len]
    x = tf.squeeze(tf.matmul(alpha, x), axis=1) # [bs, dim]
    x = tf.nn.dropout(x, 1.0 - dropout)
    return x

def output_attention(c, q, filters=128 * 3, name=None):
    q = tf.expand_dims(
        dense(q, filters, name=name, reuse=tf.AUTO_REUSE, kernel_initializer=initializer, kernel_regularizer=regularizer),
        axis=-1)  # [bs, dim, 1]
    cq = tf.squeeze(tf.matmul(c, q), axis=-1) # [bs, c_len]
    return cq

def GRUCell(x, h, hidden_size=128, filters=256):
    with tf.variable_scope('Output_GRU_Cell', reuse=tf.AUTO_REUSE):
        gate_kernel = tf.get_variable("gate_kernel",
                                      shape=(hidden_size + filters, 2 * hidden_size),
                                      dtype=tf.float32,
                                      initializer=initializer,
                                      regularizer=regularizer)
        gate_bias = tf.get_variable("gates_bias",
                                    shape=(2 * hidden_size),
                                    dtype=tf.float32,
                                    regularizer=regularizer)
        candidate_kernel = tf.get_variable("candidate_kernel",
                                           shape=(hidden_size + filters, hidden_size),
                                           dtype=tf.float32,
                                           initializer=initializer,
                                           regularizer=regularizer)
        candidate_bias = tf.get_variable("candidate_bias",
                                         shape=(hidden_size),
                                         dtype=tf.float32,
                                         regularizer=regularizer)

        gate_inputs = tf.matmul(tf.concat([x, h], 1), gate_kernel) + gate_bias
        value = tf.nn.sigmoid(gate_inputs)
        r, u = tf.split(value=value, num_or_size_splits=2, axis=1)
        r_state = r * h
        candidate = tf.matmul(tf.concat([x, r_state], 1), candidate_kernel) + candidate_bias
        c = tf.nn.tanh(candidate)
        new_h = u * h + (1 - u) * c
    return new_h

def SAN(cont, ques, q_mask, name, num_turn=5, hidden_size=128, filters=128*3, dropout=0.3):
    with tf.variable_scope(name):
        ques_mem = linear_sum_attention(ques, q_mask, dropout)
        start_scores_list = []
        end_scores_list = []
        for turn in range(num_turn):
            st_scores = output_attention(cont, ques_mem, filters=filters, name='start_output') # [bs, c_len]
            start_scores_list.append(st_scores)
            end_scores = output_attention(cont, ques_mem, filters=filters, name='end_output') # [bs, c_len]
            end_scores_list.append(end_scores)
            x = tf.squeeze(tf.matmul(tf.expand_dims(tf.nn.softmax(end_scores), axis=1), cont), axis=1) # [bs, 1, c_len]->[bs, 1, dim]->[bs, dim]
            ques_mem = tf.nn.dropout(ques_mem, 1.0 - dropout)
            ques_mem = GRUCell(x, ques_mem, hidden_size=hidden_size, filters=filters) # [bs, c_len]

    return start_scores_list, end_scores_list