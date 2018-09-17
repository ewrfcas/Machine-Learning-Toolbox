import tensorflow as tf
import os
import logging
logging.getLogger().setLevel(logging.INFO)


os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'


def model_fn(features, labels, mode):
    layer = tf.layers.Dense(1)
    logits = layer(features)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {"logits": logits}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.mean_squared_error(labels=labels, predictions=tf.squeeze(logits))

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def input_fn():
    features = tf.data.Dataset.from_tensors([[1.]]).repeat(100).batch(10)
    labels = tf.data.Dataset.from_tensors(1.).repeat(100).batch(10)
    return tf.data.Dataset.zip((features, labels))

distribution = tf.contrib.distribute.MirroredStrategy(num_gpus=2)
config = tf.estimator.RunConfig(train_distribute=distribution)
classifier = tf.estimator.Estimator(model_fn=model_fn, config=config, model_dir='model')
classifier.train(input_fn=input_fn)
classifier.evaluate(input_fn=input_fn)