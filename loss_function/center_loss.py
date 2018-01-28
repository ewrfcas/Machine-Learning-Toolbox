# tensorflow version
import tensorflow as tf

# example:
# https://github.com/godfanmiao/MNIST_CNN_CENTERLOSS_TENSORFLOW/blob/master/MNIST_CNN_BN_CENTERLOSS.ipynb

def center_loss(features, label, alpha, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    # 获取特征向量长度
    nrof_features = features.get_shape()[1]

    # 生成可以共享的变量centers
    with tf.variable_scope('center', reuse=True):
        centers = tf.get_variable('centers')
    label = tf.reshape(label, [-1])

    # 取出对应label下对应的center值，注意label里面的值可能会重复，因为一个标签下有可能会出现多个人
    centers_batch = tf.gather(centers, label)

    # 求特征点到中心的距离并乘以一定的系数，alfa是center的更新速度，越大代表更新的越慢
    diff = centers_batch - features

    # 获取一个batch中同一样本出现的次数，这里需要理解论文中的更新公式
    unique_label, unique_idx, unique_count = tf.unique_with_counts(label)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    # 更新center，输出是将对应于label的centers减去对应的diff，如果同一个标签出现多次，那么就减去多次
    centers = tf.scatter_sub(centers, label, diff)

    # 求center loss，这里是将l2_loss里面的值进行平方相加，再除以2，并没有进行开方
    loss = tf.nn.l2_loss(features - centers_batch)
    return loss, centers

# keras version by@苏god
from keras.layers import *
from keras.models import *
import numpy as np

nb_classes = 100 # 类别数，同时也是聚类中心数
feature_size = 32 # 聚类的deep feature维度

input_image = Input(shape=(224,224,3))
cnn = Conv2D(10, (2,2))(input_image)
cnn = MaxPooling2D((2,2))(cnn)
cnn = Flatten()(cnn)
feature = Dense(feature_size, activation='relu')(cnn)
predict = Dense(nb_classes, activation='softmax', name='softmax')(feature) #至此，得到一个常规的softmax分类模型

input_target = Input(shape=(1,))
centers = Embedding(nb_classes, feature_size)(input_target) # Embedding层用来存放中心(Embedding中的weights矩阵为nb_classes*feature_size,每行是一个中心)
l2_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:,0]), 1, keepdims=True), name='l2_loss')([feature,centers])

model_train = Model(inputs=[input_image,input_target], outputs=[predict,l2_loss])
model_train.compile(optimizer='adam', loss=['sparse_categorical_crossentropy',lambda y_true,y_pred: y_pred], loss_weights=[1.,0.2], metrics={'softmax':'accuracy'})

model_predict = Model(inputs=input_image, outputs=predict)
model_predict.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

train_images=np.random.random((500,224,224,3))
train_targets=np.random.randint(0,100,500) # 由于Embedding的存在，无需one-hot
random_y=np.random.random(500) # 由于keras必须要有y_true，这里其实是无用的
model_train.fit([train_images,train_targets], [train_targets,random_y], epochs=10)