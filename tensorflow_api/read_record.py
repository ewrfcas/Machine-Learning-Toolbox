import tensorflow as tf

def get_record_parser(is_test=False):
    def parser(example):
        para_limit = 400 if is_test else 128
        ques_limit = 400 if is_test else 128
        char_limit = 64
        features = tf.parse_single_example(example,
                                           features={
                                               "context_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_idxs": tf.FixedLenFeature([], tf.string),
                                               "context_char_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_char_idxs": tf.FixedLenFeature([], tf.string),
                                               'feats': tf.FixedLenFeature([], tf.string),
                                               "y1": tf.FixedLenFeature([], tf.float32),
                                               "y2": tf.FixedLenFeature([], tf.float32),
                                               "id": tf.FixedLenFeature([], tf.int64)
                                           })
        context_idxs = tf.reshape(tf.decode_raw(features["context_idxs"], tf.int64), [para_limit])
        ques_idxs = tf.reshape(tf.decode_raw(features["ques_idxs"], tf.int64), [ques_limit])
        context_char_idxs = tf.reshape(tf.decode_raw(features["context_char_idxs"], tf.int64), [para_limit, char_limit])
        ques_char_idxs = tf.reshape(tf.decode_raw(features["ques_char_idxs"], tf.int64), [ques_limit, char_limit])
        feats = tf.reshape(tf.decode_raw(features['feats'], tf.float64), [para_limit])
        y1 = features["y1"]
        y2 = features["y2"]
        qa_id = features["id"]

        return context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, feats, y1, y2, qa_id

    return parser



dataset = tf.data.TFRecordDataset('train.tfrecords').map(get_record_parser(), num_parallel_calls=2).shuffle(15000)
dataset = dataset.batch(32)
# GPU->CPU->GPU
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    for i in range(20):
        sess.run(iterator.initializer)
        print('epoch:',i+1)
        while True:
            try:
                context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, feats, y1, y2, qa_id = sess.run(next_element)
                print(qa_id)
            except tf.errors.OutOfRangeError:
                print("End of epoch %d" % (i+1))
                break
