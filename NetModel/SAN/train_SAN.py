import tensorflow as tf
import os
import time
import numpy as np
import json
import SAN_model as SAN
import tensorflow.contrib.slim as slim
import util
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def cal_ETA(t_start, i, n_batch):
    t_temp = time.time()
    t_avg = float(int(t_temp) - int(t_start)) / float(i + 1)
    if n_batch - i - 1 > 0:
        return int((n_batch - i - 1) * t_avg)
    else:
        return int(t_temp) - int(t_start)


data_source = '../QANet_tf/dataset_pre'

config = {
    'char_dim': 64,
    'cont_limit': 400,
    'ques_limit': 50,
    'char_limit': 16,
    'ans_limit': -1,
    'filters': 256,
    'dropout': 0.2,
    'dropout_emb': 0.15,
    'dropout_att': 0.2,
    'dropout_rnn': 0.0,
    'l2_norm': 3e-7,
    'decay': 0.999,
    'gamma': 1.0,
    'learning_rate': 2e-3,
    'grad_clip': 5.0,
    'use_elmo': 0,
    'use_cove': 1,
    'optimizer': 'adam',
    'cove_path': 'Keras_CoVe_2layers.h5',
    'train_tfrecords': '../QANet_tf/tfrecords/train_pre_elmo_cove2layers.tfrecords',
    'dev_tfrecords': '../QANet_tf/tfrecords/dev_pre_elmo_cove2layers.tfrecords',
    'batch_size': 32,
    'epoch': 30,
    'origin_path': None,  # not finetune
    'path': 'SANV000'
}


def get_record_parser(is_test=False):
    def parser(example):
        para_limit = 1000 if is_test else config['cont_limit']
        ques_limit = 50 if is_test else config['ques_limit']
        char_limit = config['char_limit']
        features = tf.parse_single_example(example,
                                           features={
                                               "context_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_idxs": tf.FixedLenFeature([], tf.string),
                                               "context_char_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_char_idxs": tf.FixedLenFeature([], tf.string),
                                               'elmo_context_feat': tf.FixedLenFeature([], tf.string),
                                               'elmo_question_feat': tf.FixedLenFeature([], tf.string),
                                               'cove_context_feat_low': tf.FixedLenFeature([], tf.string),
                                               'cove_context_feat_high': tf.FixedLenFeature([], tf.string),
                                               'cove_question_feat_low': tf.FixedLenFeature([], tf.string),
                                               'cove_question_feat_high': tf.FixedLenFeature([], tf.string),
                                               "y1": tf.FixedLenFeature([], tf.string),
                                               "y2": tf.FixedLenFeature([], tf.string),
                                               "y1p": tf.FixedLenFeature([], tf.string),
                                               "y2p": tf.FixedLenFeature([], tf.string),
                                               "qid": tf.FixedLenFeature([], tf.int64)
                                           })
        context_idxs = tf.reshape(tf.decode_raw(features["context_idxs"], tf.int32), [para_limit])
        ques_idxs = tf.reshape(tf.decode_raw(features["ques_idxs"], tf.int32), [ques_limit])
        context_char_idxs = tf.reshape(tf.decode_raw(features["context_char_idxs"], tf.int32), [para_limit, char_limit])
        ques_char_idxs = tf.reshape(tf.decode_raw(features["ques_char_idxs"], tf.int32), [ques_limit, char_limit])
        elmo_context_feat = tf.reshape(tf.decode_raw(features['elmo_context_feat'], tf.float32), [-1, 3, 1024])
        elmo_question_feat = tf.reshape(tf.decode_raw(features['elmo_question_feat'], tf.float32), [-1, 3, 1024])
        cove_context_feat_low = tf.reshape(tf.decode_raw(features['cove_context_feat_low'], tf.float32), [para_limit, 600])
        cove_context_feat_high = tf.reshape(tf.decode_raw(features['cove_context_feat_high'], tf.float32), [para_limit, 600])
        cove_question_feat_low = tf.reshape(tf.decode_raw(features['cove_question_feat_low'], tf.float32), [ques_limit, 600])
        cove_question_feat_high = tf.reshape(tf.decode_raw(features['cove_question_feat_high'], tf.float32), [ques_limit, 600])
        y1 = tf.reshape(tf.decode_raw(features["y1"], tf.int32), [para_limit + 1])
        y2 = tf.reshape(tf.decode_raw(features["y2"], tf.int32), [para_limit + 1])
        y1p = tf.reshape(tf.decode_raw(features["y1p"], tf.int32), [para_limit])
        y2p = tf.reshape(tf.decode_raw(features["y2p"], tf.int32), [para_limit])
        # qid = features["qid"]
        return context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, \
               elmo_context_feat, elmo_question_feat, \
               cove_context_feat_low, cove_context_feat_high, cove_question_feat_low, cove_question_feat_high, \
               y1, y2, y1p, y2p

    return parser


# loading data
val_qid = np.load(data_source + '/dev_qid.npy').astype(np.int32)
with open(data_source + '/test_eval.json', "r") as fh:
    eval_file = json.load(fh)

train_dataset = tf.data.TFRecordDataset(config['train_tfrecords']) \
    .map(get_record_parser(is_test=False), num_parallel_calls=4) \
    .shuffle(15000) \
    .padded_batch(config['batch_size'], padded_shapes=([None],
                                                       [None],
                                                       [None, None],
                                                       [None, None],
                                                       [None, None, None],
                                                       [None, None, None],
                                                       [None, None],
                                                       [None, None],
                                                       [None, None],
                                                       [None, None],
                                                       [None],
                                                       [None],
                                                       [None],
                                                       [None]))
train_iterator = train_dataset.make_initializable_iterator()
train_next_element = train_iterator.get_next()
train_sum = 129941

dev_dataset = tf.data.TFRecordDataset(config['dev_tfrecords']) \
    .map(get_record_parser(is_test=False), num_parallel_calls=4) \
    .padded_batch(config['batch_size'], padded_shapes=([None],
                                                       [None],
                                                       [None, None],
                                                       [None, None],
                                                       [None, None, None],
                                                       [None, None, None],
                                                       [None, None],
                                                       [None, None],
                                                       [None, None],
                                                       [None, None],
                                                       [None],
                                                       [None],
                                                       [None],
                                                       [None]))
dev_iterator = dev_dataset.make_initializable_iterator()
dev_next_element = dev_iterator.get_next()
dev_sum = 11730

# load embedding matrix
word_mat = np.load(data_source + '/word_emb_mat.npy')
char_mat = np.load(data_source + '/char_emb_mat.npy')
print('load embedding over...')

model = SAN.Model(config, word_mat=word_mat, char_mat=char_mat)
print('model init over...')
sess_config = tf.ConfigProto(allow_soft_placement=True)
sess_config.gpu_options.allow_growth = True
best_f1 = 0
best_em = 0
f1s = []
ems = []

with tf.Session(config=sess_config) as sess:
    sess.run(tf.global_variables_initializer())
    # scope with trainable weights
    variables_to_restore = slim.get_variables_to_restore(include=['Input_Embedding_Mat',
                                                                  'Input_Embedding_Layer',
                                                                  'Encoder_Layers',
                                                                  'Point_Network',
                                                                  'EMA_Weights'])
    saver = tf.train.Saver(variables_to_restore, max_to_keep=10)
    if config['origin_path'] is not None and os.path.exists(os.path.join('model', config['origin_path'], 'checkpoint')):
        saver.restore(sess, tf.train.latest_checkpoint(os.path.join('model', str(config['origin_path']) + '/')))

    for i_epoch in range(config['epoch']):
        sess.run(train_iterator.initializer)
        i_batch = 0
        train_n_batch = train_sum // config['batch_size'] + 1
        val_n_batch = dev_sum // config['batch_size'] + 1
        if i_epoch+1>=10 and (i_epoch+1)%10==0:
            config['learning_rate']*=0.5
        t_start = time.time()
        last_train_str = "\r"
        sum_loss = 0
        while True:
            try:
                if i_batch==1:
                    t_start = time.time()
                context_idxs, ques_idxs, \
                context_char_idxs, ques_char_idxs, \
                elmo_context_feat, elmo_question_feat, \
                cove_context_feat_low, cove_context_feat_high, cove_question_feat_low, cove_question_feat_high, \
                y1, y2, y1p, y2p = sess.run(train_next_element)
                feed_dict_ = {model.contw_input_: context_idxs, model.quesw_input_: ques_idxs,
                              model.contc_input_: context_char_idxs, model.quesc_input_: ques_char_idxs,
                              model.y_start_: y1, model.y_end_: y2,
                              # model.yp_start_: y1p, model.yp_end_: y2p,
                              model.un_size: context_idxs.shape[0],
                              model.dropout: config['dropout'],
                              model.dropout_emb: config['dropout_emb'],
                              model.dropout_att: config['dropout_att'],
                              model.dropout_rnn: config['dropout_rnn'],
                              model.learning_rate: config['learning_rate']}
                # if config['use_elmo']!=0:
                #     feed_dict_[model.elmo_cont]=elmo_context_feat
                #     feed_dict_[model.elmo_ques]=elmo_question_feat
                if config['use_cove']==1:
                    feed_dict_[model.cove_cont_low_]=cove_context_feat_low
                    feed_dict_[model.cove_cont_high_] = cove_context_feat_high
                    feed_dict_[model.cove_ques_low_]=cove_question_feat_low
                    feed_dict_[model.cove_ques_high_] = cove_question_feat_high
                loss_value, _ = sess.run([model.loss, model.ema_train_op], feed_dict=feed_dict_)
                sum_loss += loss_value
                last_train_str = "\r[epoch:%d/%d, steps:%d/%d] -ETA:%ds -loss:%.4f" % (
                    i_epoch + 1, config['epoch'], i_batch + 1, train_n_batch, cal_ETA(t_start, i_batch, train_n_batch),
                    sum_loss / (i_batch+1))
                print(last_train_str, end='      ', flush=True)
                i_batch += 1
            except tf.errors.OutOfRangeError:
                break

        # validating step
        # # save the temp weights and do ema
        # if config['decay'] < 1.0:
        #     saver.save(sess, os.path.join('model', config['path'], 'temp_model.ckpt'))
        #     sess.run(model.assign_vars)
        #     print('EMA over...')
        sess.run(dev_iterator.initializer)
        print('\n')
        sum_loss_val = 0
        y1s = []
        y2s = []
        last_val_str = "\r"
        i_batch = 0
        while True:
            try:
                context_idxs, ques_idxs, \
                context_char_idxs, ques_char_idxs, \
                elmo_context_feat, elmo_question_feat, \
                cove_context_feat_low, cove_context_feat_high, cove_question_feat_low, cove_question_feat_high, \
                y1, y2, y1p, y2p = sess.run(dev_next_element)
                feed_dict_ = {model.contw_input_: context_idxs, model.quesw_input_: ques_idxs,
                              model.contc_input_: context_char_idxs, model.quesc_input_: ques_char_idxs,
                              model.y_start_: y1, model.y_end_: y2,
#                               model.yp_start_: y1p, model.yp_end_: y2p,
                              model.un_size: context_idxs.shape[0]}
                # if config['use_elmo']!=0:
                #     feed_dict_[model.elmo_cont]=elmo_context_feat
                #     feed_dict_[model.elmo_ques]=elmo_question_feat
                if config['use_cove']==1:
                    feed_dict_[model.cove_cont_low_] = cove_context_feat_low
                    feed_dict_[model.cove_cont_high_] = cove_context_feat_high
                    feed_dict_[model.cove_ques_low_] = cove_question_feat_low
                    feed_dict_[model.cove_ques_high_] = cove_question_feat_high
                loss_value, y1, y2 = sess.run([model.loss, model.mask_output1, model.mask_output2],
                                              feed_dict=feed_dict_)
                y1s.append(y1)
                y2s.append(y2)
                sum_loss_val += loss_value
                last_val_str = "\r[validate:%d/%d] -loss:%.4f" % (
                    i_batch + 1, val_n_batch, sum_loss_val / (i_batch + 1))
                print(last_val_str, end='      ', flush=True)
                i_batch += 1
            except tf.errors.OutOfRangeError:
                y1s = np.concatenate(y1s)
                y2s = np.concatenate(y2s)
                answer_dict, _, noanswer_num = util.convert_tokens(eval_file, val_qid.tolist(), y1s.tolist(),
                                                                   y2s.tolist(), data_type=2)
                metrics = util.evaluate(eval_file, answer_dict)
                ems.append(metrics['exact_match'])
                f1s.append(metrics['f1'])
                saver.save(sess, os.path.join('model', config['path'], 'model.ckpt'),
                           global_step=(i_epoch + 1) * train_n_batch)
                print(last_val_str,
                      " -EM:%.2f%%, -F1:%.2f%% -Noanswer:%d" % (
                          metrics['exact_match'], metrics['f1'], noanswer_num),
                      end=' ', flush=True)
                print('\n')
                result = pd.DataFrame([ems, f1s], index=['em', 'f1']).transpose()
                result.to_csv('log/result_' + config['path'] + '.csv', index=None)

                # # recover the model
                # if config['decay'] < 1.0:
                #     saver.restore(sess, os.path.join('model', config['path'], 'temp_model.ckpt'))
                #     print('recover weights over...')
                break
