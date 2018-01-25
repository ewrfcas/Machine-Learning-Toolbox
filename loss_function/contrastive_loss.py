import tensorflow as tf

# contrastive loss
# loss=(1/2N)*\sum(y*d^2+(1-y)max(alpha-d,0)^2)
def contrastive_loss(y_true, y_pred, alpha = 0.2):
    left = y_pred[0]
    right = y_pred[1]
    d = tf.sqrt(tf.reduce_sum(tf.square(left - right), 1))

    # if label(left)==label(right) y_same=1
    # else y_same=0
    y_same = tf.cast(tf.equal(y_true[0],y_true[1]),tf.float32)
    loss = y_same * tf.square(d) + (1-y_same) * tf.square(tf.maximum(alpha - d, 0.))
    loss = 0.5 * tf.reduce_mean(loss)

    return loss