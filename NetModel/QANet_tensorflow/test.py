import numpy as np
import QANet_model
import QANet_model_alter
import tensorflow as tf
import util
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '3'


def training_shuffle(data, seed=None):
    if seed is not None:
        np.random.seed(seed)
    index = np.arange(data[0].shape[0])
    np.random.shuffle(index)
    for i, d in enumerate(data):
        data[i] = data[i][index, ::]
    return data


def next_batch(data, batch_size, iteration):
    data_temp = []
    start_index = iteration * batch_size
    if iteration == (data[0].shape[0] // batch_size) - 1:
        end_index = -1
    else:
        end_index = (iteration + 1) * batch_size
    for i, d in enumerate(data):
        if end_index != -1:
            data_temp.append(data[i][start_index: end_index, ::])
        else:
            data_temp.append(data[i][-1*batch_size:, ::])
    return data_temp


# load testset
test_context_word=np.load('dataset/test_contw_input.npy')
test_question_word=np.load('dataset/test_quesw_input.npy')
test_context_char=np.load('dataset/test_contc_input.npy')
test_question_char=np.load('dataset/test_quesc_input.npy')
test_start_label=np.load('dataset/test_y_start.npy')
test_end_label=np.load('dataset/test_y_end.npy')
test_qid=np.load('dataset/test_qid.npy').astype(np.int32)
context_string = np.load('dataset/test_contw_strings.npy')
ques_string = np.load('dataset/test_quesw_strings.npy')
with open('dataset/test_eval.json', "r") as fh:
    eval_file = json.load(fh)

# load embedding matrix
word_mat = np.load('dataset/word_emb_mat.npy')
char_mat = np.load('dataset/char_emb_mat.npy')

test_set = [test_context_word, test_question_word, test_context_char, test_question_char, context_string, ques_string, test_start_label, test_end_label]

config = {
    'char_dim': 64,
    'cont_limit': 400,
    'ques_limit': 50,
    'char_limit': 16,
    'ans_limit': 30,
    'filters': 128,
    'num_heads': 1,
    'dropout': 0.1,
    'l2_norm': 3e-7,
    'decay': 0.9999,
    'learning_rate': 1e-3,
    'grad_clip': 5.0,
    'batch_size': 32,
    'epoch': 30,
    'path': 'QANetV201'
}

model = QANet_model.Model(config, word_mat=word_mat, char_mat=char_mat)
sess_config = tf.ConfigProto(allow_soft_placement=True)
sess_config.gpu_options.allow_growth = True

with tf.Session(config=sess_config) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(os.path.join('model',str(config['path'])+'/')))
    if config['decay'] < 1.0:
        sess.run(model.assign_vars)
    n_batch_test = test_context_word.shape[0] // config['batch_size']

    # test step
    sum_loss_val = 0
    y1s = []
    y2s = []
    last_test_str = '\r'
    for i in range(n_batch_test):
        contw_input, quesw_input, contc_input, quesc_input, contw_string, quesw_string, y_start, y_end \
            = next_batch(test_set, config['batch_size'], i)
        loss_value, y1, y2 = sess.run([model.loss, model.output1, model.output2],
                                      feed_dict={model.contw_input_: contw_input, model.quesw_input_: quesw_input,
                                                 model.contc_input_: contc_input, model.quesc_input_: quesc_input,
                                                 model.contw_strings: contw_string, model.quesw_strings: quesw_string,
                                                 model.y_start_: y_start, model.y_end_: y_end})
        y1s.append(y1)
        y2s.append(y2)
        sum_loss_val += loss_value
        last_test_str = "\r[test:%d/%d] -loss: %.4f" % (i + 1, n_batch_test, sum_loss_val / (i + 1))
        print(last_test_str, end='      ', flush=True)
    y1s = np.concatenate(y1s)
    y2s = np.concatenate(y2s)
    answer_dict, _, noanswer_num = util.convert_tokens(eval_file, test_qid.tolist(), y1s.tolist(), y2s.tolist())
    metrics = util.evaluate(eval_file, answer_dict)
    print(last_test_str,
          " -EM: %.2f%%, -F1: %.2f%% -Noanswer: %d" % (metrics['exact_match'], metrics['f1'], noanswer_num), end=' ',
          flush=True)
    print('\n')