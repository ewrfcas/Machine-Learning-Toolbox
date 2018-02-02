import pandas as pd

mnist_train = pd.read_csv('../data/mnist_train.csv',header=None)
mnist_test = pd.read_csv('../data/mnist_test.csv',header=None)

mnist_train.head()
X_train=mnist_train.loc[:,1:].values
y_train=mnist_train.loc[:,0].values
X_test=mnist_test.loc[:,1:].values
y_test=mnist_test.loc[:,0].values

X_train=X_train/255.
X_test=X_test/255.

print('X_train:',X_train.shape)
print('y_train:',y_train.shape)
print('X_test:',X_test.shape)
print('y_test:',y_test.shape)

import tensorflow as tf
sess = tf.InteractiveSession()

inputs = tf.placeholder(tf.float32, [None, 784])
inputs_reshaped = tf.reshape(inputs, (-1,28,28,1))
y = tf.placeholder(tf.int64, [None, 1])
y_onehot = tf.one_hot(y, 10)
n_class=10
n_feature=512


# 定义模型，复杂loss
from tensorflow.contrib.keras import layers


def VGG6(inputs, n_class=10):
    # Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(512, activation='relu', name='fc1')(x)
    features = layers.Dense(512, activation='relu', name='fc2')(x)
    outputs = layers.Dense(n_class, activation='softmax', name='predictions')(features)

    return outputs

# training graph
import numpy as np

# cal loss
preds = VGG6(inputs_reshaped)
softmax_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=y_onehot))

# cal acc
result = tf.argmax(preds, 1)
ground_truth = tf.reshape(y, [-1])
correct_prediction = tf.equal(result, ground_truth)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# train
train_step = tf.train.AdamOptimizer(1e-4).minimize(softmax_loss)

n_epochs = 50
batch_size = 128
n_batch = int(X_train.shape[0] / batch_size)
n_batch_val = int(X_test.shape[0] / batch_size)
sess.run(tf.global_variables_initializer())
early_stop=10
best_val_acc=0
patience=0
for epoch in range(n_epochs):
    # training step
    index = np.arange(X_train.shape[0])
    np.random.shuffle(index)
    X_train = X_train[index, :]
    y_train = y_train[index]
    sum_loss = 0
    sum_acc = 0
    last_train_str = ""
    for i in range(n_batch):
        x_temp = X_train[i * batch_size:(i + 1) * batch_size, :]
        y_temp = y_train[i * batch_size:(i + 1) * batch_size]
        y_temp = np.reshape(y_temp, (batch_size, 1))
        feed_dict = {inputs: x_temp, y: y_temp}
        _, loss_value, acc_value = sess.run([train_step, softmax_loss, accuracy], feed_dict=feed_dict)
        sum_loss += loss_value
        sum_acc += (acc_value * 100)
        last_train_str = "\r[epoch:%d/%d, steps:%d/%d] -loss: %.4f - acc: %.2f%%" % \
                         (epoch + 1, n_epochs, i + 1, n_batch, sum_loss / (i + 1), sum_acc / (i + 1))
        print(last_train_str, end='      ', flush=True)

    # validating step
    sum_loss = 0
    sum_acc = 0
    for i in range(n_batch_val):
        x_temp = X_test[i * batch_size:(i + 1) * batch_size, :]
        y_temp = y_test[i * batch_size:(i + 1) * batch_size]
        y_temp = np.reshape(y_temp, (batch_size, 1))
        feed_dict = {inputs: x_temp, y: y_temp}
        loss_value, acc_value = sess.run([softmax_loss, accuracy], feed_dict=feed_dict)
        sum_loss += loss_value
        sum_acc += (acc_value * 100)
        print(last_train_str + "  [validate:%d/%d] -loss: %.4f - acc: %.2f%%" % \
              (i + 1, n_batch_val, sum_loss / (i + 1), sum_acc / (i + 1)), \
              end='      ', flush=True)
    print('\n')

    # early stop
    if sum_acc / (n_batch_val) > best_val_acc:
        patience = 0
        print(str(round(sum_acc, 2)) + '% > ' + str(round(best_val_acc, 2)) + '%', 'patience:', patience)
        best_val_acc = sum_acc / (n_batch_val)
    else:
        patience += 1
        print(str(round(sum_acc, 2)) + '% <= ' + str(round(best_val_acc, 2)) + '%', 'patience:', patience)
    if patience > early_stop:
        print('early stopping!')
        break

