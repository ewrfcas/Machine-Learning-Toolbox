from keras import initializers, regularizers, constraints
import keras.backend as K
from keras.engine.topology import Layer

class Attention(Layer):
    def __init__(self, timesteps, bias=True, simple=False,
                 W_regularizer=None, W_constraint=None,
                 V_regularizer=None, V_constraint=None,
                 **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.bias = bias
        self.timesteps = timesteps
        self.simple = simple
        self.W_regularizer = regularizers.get(W_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.V_regularizer = regularizers.get(V_regularizer)
        self.V_constraint = constraints.get(V_constraint)

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.features_dim = input_shape[-1]
        self.W = self.add_weight((self.timesteps, self.timesteps),
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint,
                                 name='{}_W'.format(self.name))
        if not self.simple:
            self.V = self.add_weight((self.features_dim, 1),
                                     initializer=self.init,
                                     regularizer=self.V_regularizer,
                                     constraint=self.V_constraint,
                                     name='{}_V'.format(self.name))

        if self.bias:
            self.b = self.add_weight((self.timesteps,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name))
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        x_transpose = K.permute_dimensions(x, (0,2,1))
        e = K.dot(x_transpose, self.W)
        if self.bias:
            e += self.b
        e = K.tanh(e)
        if not self.simple:
            e = K.permute_dimensions(e, (0,2,1))
            e = K.reshape(K.dot(e, self.V), (-1, self.timesteps))
        else:
            e = K.mean(e, axis=1)
        a = K.exp(e)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a_weights = a / K.cast(K.sum(a, axis=-1, keepdims=True) + K.epsilon(), K.floatx())
        weighted_output = x * K.expand_dims(a_weights, axis=-1)

        return [K.sum(weighted_output, axis=1), a_weights]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.features_dim), (input_shape[0], self.timesteps)]