import numpy as np
import NetModel.QANet_tensorflow.QANet_model as QANet_model
import tensorflow as tf
import NetModel.QANet_tensorflow.util as util
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


def next_batch(data, batch_size, iteration, test=False):
    data_temp = []
    start_index = iteration * batch_size
    if iteration == (data[0].shape[0] // batch_size) - 1 and test:
        end_index = -1
    else:
        end_index = (iteration + 1) * batch_size
    for i, d in enumerate(data):
        if end_index != -1:
            data_temp.append(data[i][start_index: end_index, ::])
        else:
            data_temp.append(data[i][start_index:, ::])
    return data_temp


# load testset
test_context_word=np.load('../QANet/dataset2/test_contw_input.npy')
test_question_word=np.load('../QANet/dataset2/test_quesw_input.npy')
test_context_char=np.load('../QANet/dataset2/test_contc_input.npy')
test_question_char=np.load('../QANet/dataset2/test_quesc_input.npy')
test_start_label=np.load('../QANet/dataset2/test_y_start.npy')
test_end_label=np.load('../QANet/dataset2/test_y_end.npy')
test_qid=np.load('../QANet/dataset2/test_qid.npy').astype(np.int32)
with open('../QANet/dataset2/test_eval.json', "r") as fh:
    eval_file = json.load(fh)

# load embedding matrix
word_mat = np.load('../QANet/dataset2/word_emb_mat2.npy')
char_mat = np.load('../QANet/dataset2/char_emb_mat2.npy')

test_set = [test_context_word, test_question_word, test_context_char, test_question_char, test_start_label, test_end_label]

config = {
    'char_dim': 64,
    'cont_limit': 400,
    'ques_limit': 50,
    'char_limit': 16,
    'ans_limit': 50,
    'filters': 128,
    'num_heads': 1,
    'dropout': 0.1,
    'l2_norm': 3e-7,
    'decay': 0.9999,
    'learning_rate': 1e-3,
    'grad_clip': 5.0,
    'batch_size': 32,
    'epoch': 25,
    'path': 'QANetV100'
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
        contw_input, quesw_input, contc_input, quesc_input, y_start, y_end \
            = next_batch(test_set, config['batch_size'], i, test=True)
        loss_value, y1, y2 = sess.run([model.loss, model.output1, model.output2],
                                      feed_dict={model.contw_input: contw_input, model.quesw_input: quesw_input,
                                                 model.contc_input: contc_input, model.quesc_input: quesc_input,
                                                 model.y_start: y_start, model.y_end: y_end})
        y1s.append(y1)
        y2s.append(y2)
        sum_loss_val += loss_value
        last_test_str = "[test:%d/%d] -loss: %.4f" % (i + 1, n_batch_test, sum_loss_val / (i + 1))
        print(last_test_str, end='      ', flush=True)
    y1s = np.concatenate(y1s)
    y2s = np.concatenate(y2s)
    answer_dict, remapped_dict = util.convert_tokens(eval_file, test_qid.tolist(), y1s.tolist(), y2s.tolist())
    metrics = util.evaluate(eval_file, answer_dict)
    print(last_test_str, " -EM: %.2f%%, -F1: %.2f%%" % (metrics['exact_match'], metrics['f1']), end=' ', flush=True)
    print('\n')


