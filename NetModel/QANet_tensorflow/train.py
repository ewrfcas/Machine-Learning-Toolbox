import numpy as np
import pandas as pd
import QANet_model
import tensorflow as tf
import util
import json
import os
import time

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

def cal_ETA(t_start, i, n_batch):
    t_temp = time.time()
    t_avg = float(int(t_temp) - int(t_start)) / float(i + 1)
    if n_batch - i - 1 > 0:
        return int((n_batch - i - 1) * t_avg)
    else:
        return int(t_temp)-int(t_start)


# load trainset
context_word = np.load('../QANet/dataset2/train_contw_input.npy').astype(np.int32)
question_word = np.load('../QANet/dataset2/train_quesw_input.npy').astype(np.int32)
context_char = np.load('../QANet/dataset2/train_contc_input.npy').astype(np.int32)
question_char = np.load('../QANet/dataset2/train_quesc_input.npy').astype(np.int32)
start_label = np.load('../QANet/dataset2/train_y_start.npy').astype(np.int32)
end_label = np.load('../QANet/dataset2/train_y_end.npy').astype(np.int32)

# load valset
val_context_word = np.load('../QANet/dataset2/dev_contw_input.npy').astype(np.int32)
val_question_word = np.load('../QANet/dataset2/dev_quesw_input.npy').astype(np.int32)
val_context_char = np.load('../QANet/dataset2/dev_contc_input.npy').astype(np.int32)
val_question_char = np.load('../QANet/dataset2/dev_quesc_input.npy').astype(np.int32)
val_start_label = np.load('../QANet/dataset2/dev_y_start.npy').astype(np.int32)
val_end_label = np.load('../QANet/dataset2/dev_y_end.npy').astype(np.int32)
val_qid = np.load('../QANet/dataset2/dev_qid.npy').astype(np.int32)
with open('../QANet/dataset2/dev_eval.json', "r") as fh:
    eval_file = json.load(fh)

# load embedding matrix
word_mat = np.load('../QANet/dataset2/word_emb_mat2.npy')
char_mat = np.load('../QANet/dataset2/char_emb_mat2.npy')

train_set = [context_word, question_word, context_char, question_char, start_label, end_label]
val_set = [val_context_word, val_question_word, val_context_char, val_question_char, val_start_label, val_end_label]

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
    'path': 'QANetV102'
}

model = QANet_model.Model(config, word_mat=word_mat, char_mat=char_mat)
sess_config = tf.ConfigProto(allow_soft_placement=True)
sess_config.gpu_options.allow_growth = True

best_f1 = 0
best_em = 0
f1s = []
ems = []

with tf.Session(config=sess_config) as sess:
    if not os.path.exists(os.path.join('model',config['path'])):
        os.mkdir(os.path.join('model',config['path']))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    n_batch = context_word.shape[0] // config['batch_size']
    n_batch_val = val_context_word.shape[0] // config['batch_size']
    for epoch in range(config['epoch']):
        train_set = training_shuffle(train_set)
        t_start = time.time()
        last_train_str = "\r"
        # training step
        sum_loss = 0
        for i in range(n_batch):
            contw_input, quesw_input, contc_input, quesc_input, y_start, y_end \
                = next_batch(train_set, config['batch_size'], i)
            loss_value, _ = sess.run([model.loss, model.train_op],
                                     feed_dict={model.contw_input_: contw_input, model.quesw_input_: quesw_input,
                                                model.contc_input_: contc_input, model.quesc_input_: quesc_input,
                                                model.y_start_: y_start, model.y_end_: y_end})
            sum_loss += loss_value
            last_train_str = "\r[epoch:%d/%d, steps:%d/%d] -ETA: %ds -loss: %.4f" % (
            epoch + 1, config['epoch'], i + 1, n_batch, cal_ETA(t_start, i, n_batch), sum_loss / (i + 1))
            print(last_train_str, end='      ', flush=True)

        # validating step
        sum_loss_val = 0
        y1s = []
        y2s = []
        last_val_str = "\r"
        for i in range(n_batch_val):
            contw_input, quesw_input, contc_input, quesc_input, y_start, y_end \
                = next_batch(val_set, config['batch_size'], i, test=True)
            loss_value, y1, y2 = sess.run([model.loss, model.output1, model.output2],
                                          feed_dict={model.contw_input_: contw_input, model.quesw_input_: quesw_input,
                                                     model.contc_input_: contc_input, model.quesc_input_: quesc_input,
                                                     model.y_start_: y_start, model.y_end_: y_end})
            y1s.append(y1)
            y2s.append(y2)
            sum_loss_val += loss_value
            last_val_str = last_train_str + "  [validate:%d/%d] -loss: %.4f" % (
            i + 1, n_batch_val, sum_loss_val / (i + 1))
            print(last_val_str, end='      ', flush=True)
        y1s = np.concatenate(y1s)
        y2s = np.concatenate(y2s)
        answer_dict, remapped_dict = util.convert_tokens(eval_file, val_qid.tolist(), y1s.tolist(), y2s.tolist())
        metrics = util.evaluate(eval_file, answer_dict)
        ems.append(metrics['exact_match'])
        f1s.append(metrics['f1'])
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            saver.save(sess, os.path.join('model',config['path'], 'model.ckpt'), global_step=(epoch+1) * n_batch)
        print(last_val_str, " -EM: %.2f%%, -F1: %.2f%%" % (metrics['exact_match'], metrics['f1']), end=' ', flush=True)
        print('\n')

        result = pd.DataFrame([ems, f1s], index=['em', 'f1']).transpose()
        result.to_csv('log/result.csv', index=None)

    saver.save(sess, os.path.join('model',config['path'], 'model.ckpt'), global_step=config['epoch'] * n_batch)


