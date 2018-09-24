import tensorflow as tf
import numpy as np
from tqdm import tqdm

tfrecords_filename = 'train.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)
context_idxs = np.random.randint(0,1000,(3000,128))
ques_idxs = np.random.randint(0,1000,(3000,128))
context_char_idxs = np.random.randint(0,100,(3000,128,64))
ques_char_idxs = np.random.randint(0,100,(3000,128,64))
feats = np.random.random((3000,128))
y1=np.zeros(3000)
y2=np.ones(3000)
id=np.arange(3000)

for i in tqdm(range(3000)):
    record = tf.train.Example(features=tf.train.Features(feature={
                              "context_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_idxs[i,::].tostring()])),
                              "ques_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs[i,::].tostring()])),
                              "context_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_char_idxs[i,::].tostring()])),
                              "ques_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_char_idxs[i,::].tostring()])),
                              'feats': tf.train.Feature(bytes_list=tf.train.BytesList(value=[feats[i,::].tostring()])),
                              "y1": tf.train.Feature(float_list=tf.train.FloatList(value=[y1[i]])),
                              "y2": tf.train.Feature(float_list=tf.train.FloatList(value=[y2[i]])),
                              "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[id[i]]))
                              }))

    writer.write(record.SerializeToString())

writer.close()
