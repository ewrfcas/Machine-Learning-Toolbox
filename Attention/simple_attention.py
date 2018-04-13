from keras import initializers, regularizers, constraints
import keras.backend as K
from keras.engine.topology import Layer

class Attention(Layer):
    def __init__(self, timesteps, attention_size, bias=True, sparsity=False, sparsity_value = 1.,
                 W_regularizer=regularizers.l1(0.01), W_constraint=None,
                 U_regularizer=regularizers.l1(0.01), U_constraint=None,
                 **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.bias = bias
        self.timesteps = timesteps
        self.attention_size = attention_size
        self.sparsity = sparsity # make the output a more sparsity
        self.sparsity_value = sparsity_value # while sparsity_value is larger, the outputs is more sparse
        self.W_regularizer = regularizers.get(W_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.U_constraint = constraints.get(U_constraint)

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.features_dim = input_shape[-1]
        self.W = self.add_weight((self.features_dim, self.attention_size),
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint,
                                 name='{}_W'.format(self.name))

        self.U = self.add_weight((self.attention_size, 1),
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint,
                                 name='{}_U'.format(self.name))

        if self.bias:
            self.b = self.add_weight((self.attention_size,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name))
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        e = K.dot(x, self.W)
        if self.bias:
            e += self.b
        e = K.tanh(e)
        e = K.reshape(K.dot(e, self.U), (-1, self.timesteps))
        if self.sparsity:
            e = ((e - K.min(e, keepdims=True))/(K.max(e, keepdims=True) - K.min(e, keepdims=True)) - 0.5) * self.sparsity_value
        a = K.exp(e)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a_weights = a / K.cast(K.sum(a, axis=-1, keepdims=True) + K.epsilon(), K.floatx())
        weighted_output = x * K.expand_dims(a_weights, axis=-1)

        return [K.mean(weighted_output, axis=1), a_weights]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.features_dim), (input_shape[0], self.timesteps)]