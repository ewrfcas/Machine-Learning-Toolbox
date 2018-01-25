# tensorflow version
import tensorflow as tf
# example:
# https://github.com/godfanmiao/MNIST_CNN_CENTERLOSS_TENSORFLOW/blob/master/MNIST_CNN_BN_CENTERLOSS.ipynb

def island_loss(features, label, alpha, nrof_classes, lamda1=10):
    """Center loss based on the paper "Island Loss for Learning Discriminative Features in Facial Expression Recognition"
       (https://github.com/SeriaZheng/EmoNet/blob/master/loss_function/loss_paper/Island_loss.pdf)
    """
    # 获取特征向量长度
    nrof_features = features.get_shape()[1]

    # 生成可以共享的变量centers，由于center loss在计算图中只存在于一个节点处，因此这个变量只使用一次
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])

    # 取出对应label下对应的center值，注意label里面的值可能会重复，因为一个标签下有可能会出现多个人
    centers_batch = tf.gather(centers, label)

    # 求特征点到中心的距离并乘以一定的系数，diff1为center loss
    diff1 = centers_batch - features

    # 获取一个batch中同一样本出现的次数，这里需要理解论文中的更新公式
    unique_label, unique_idx, unique_count = tf.unique_with_counts(label)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff1 = diff1 / tf.cast((1 + appear_times), tf.float32)
    diff1 = alpha * diff1

    # diff2为island loss的center更新项
    diff2 = tf.zeros((nrof_classes, nrof_features))
    for i in range(nrof_classes):
        for j in range(nrof_classes):
            if i!=j:
                diff2[i, :] = tf.add((centers[i, :] / tf.sqrt(tf.reduce_sum(tf.square(centers[i, :]))) * tf.sqrt(
                    tf.reduce_sum(tf.square(centers[j, :]))))
                                     - tf.multiply((tf.reduce_sum(tf.multiply(centers[i, :], centers[j, :])) / tf.sqrt(
                    tf.reduce_sum(tf.square(centers[i, :]))) *
                                                    tf.pow(tf.sqrt(tf.reduce_sum(tf.square(centers[j, :]))), 3)),
                                                   centers[j, :]), diff2[i, :])
    diff2 = diff2 * lamda1 / (nrof_classes - 1)
    diff2 = alpha * diff2

    # 求center loss，这里是将l2_loss里面的值进行平方相加，再除以2，并没有进行开方
    loss1 = tf.nn.l2_loss(features - centers_batch)

    # 求island loss
    loss2 = tf.zeros(1)
    for i in range(nrof_classes):
        for j in range(nrof_classes):
            if i!=j:
                loss2 = tf.add(tf.add(tf.reduce_sum(tf.multiply(centers[i, :], centers[j, :])) / (
                        tf.sqrt(tf.reduce_sum(tf.square(centers[i, :]))) *
                        tf.sqrt(tf.reduce_sum(tf.square(centers[j, :])))), tf.ones(1)), loss2)
    loss2 = lamda1 * loss2

    loss = tf.add(loss1,loss2)

    # 更新center，输出是将对应于label的centers减去对应的diff，如果同一个标签出现多次，那么就减去多次(diff1与centers维度不同)
    centers = tf.scatter_sub(centers, label, diff1)
    # diff2维度与centers相同可以直接减
    centers = tf.subtract(centers, diff2)

    return loss, centers