import tensorflow as tf

# contrastive loss
# loss=(1/2N)*\sum(y*d^2+(1-y)max(alpha-d,0)^2)
def contrastive_loss(y_same, y_pred, alpha = 0.2):
    left = y_pred[0]
    right = y_pred[1]

    d = tf.reduce_sum(tf.square(left - right), 1)
    d_sqrt = tf.sqrt(d)

    loss = y_same * tf.square(tf.maximum(0., alpha - d_sqrt)) + (1 - y_same) * d #y_same=1 while label(left)==label(right) else y_same=0

    loss = 0.5 * tf.reduce_mean(loss)

    return loss