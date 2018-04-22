from keras import initializers
import keras.backend as K
from keras.engine.topology import Layer
import math
import tensorflow as tf

class ArcFaceLoss(Layer):
    def __init__(self, class_num, s=64, m=0.5, **kwargs):
        self.init = initializers.get('glorot_uniform')
        self.class_num = class_num
        self.s = s
        self.m = m
        super(ArcFaceLoss, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape[0]) == 2 and len(input_shape[1]) == 2
        self.W = self.add_weight((input_shape[0][-1], self.class_num), initializer=self.init,
                                 name='{}_W'.format(self.name))
        super(ArcFaceLoss, self).build(input_shape)

    def call(self, inputs, mask=None):
        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)
        mm = sin_m * self.m
        threshold = math.cos(math.pi - self.m)
        # inputs:
        # x: features, y_mask: 1-D or one-hot label works as mask
        x = inputs[0]
        y_mask = inputs[1]
        if y_mask.shape[-1]==1:
            y_mask = K.cast(y_mask, tf.int32)
            y_mask = K.reshape(K.one_hot(y_mask, self.class_num),(-1, self.class_num))

        # feature norm
        x = K.l2_normalize(x, axis=1)
        # weights norm
        self.W = K.l2_normalize(self.W, axis=0)

        # cos(theta+m)
        cos_theta = K.dot(x, self.W)
        cos_theta2 = K.square(cos_theta)
        sin_theta2 = 1. - cos_theta2
        sin_theta = K.sqrt(sin_theta2)
        cos_tm = self.s * ((cos_theta * cos_m) - (sin_theta * sin_m))

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - threshold
        cond = K.cast(K.relu(cond_v), dtype=tf.bool)
        keep_val = self.s * (cos_theta - mm)
        cos_tm_temp = tf.where(cond, cos_tm, keep_val)

        # mask by label
        y_mask =+ K.epsilon()
        inv_mask = 1. - y_mask
        s_cos_theta = self.s * cos_theta
        output = K.softmax((s_cos_theta * inv_mask) + (cos_tm_temp * y_mask))

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.class_num

# def model():
#     inputs=Input((112,112,3))
#     label=Input((3,))
#     x = Conv2D(16,3,activation='relu')(inputs)
#     x = MaxPooling2D()(x)
#     x = Conv2D(32,3,activation='relu')(x)
#     x = MaxPooling2D()(x)
#     x = GlobalAveragePooling2D()(x)
#     x = ArcFaceLoss(class_num=3)([x,label])
#     x = Dense(3, activation='softmax')(x)
#
#     return Model(inputs=[inputs,label], outputs=x)
#
# model=model()
# model.compile(loss='categorical_crossentropy',optimizer='adam')
# model.summary()
# X=np.random.random((1000,112,112,3))
# y=np.random.randint(0,3,(1000))
# y=utils.to_categorical(y,3)
# model.fit([X,y],y)