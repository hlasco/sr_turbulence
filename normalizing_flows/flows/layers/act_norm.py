import tensorflow as tf
from tensorflow.keras import layers

class ActNorm(layers.Layer):
    def __init__(self, name='act_norm'):
        super().__init__(name=name, dtype=tf.float32)

    def build(self, input_shape):
        self.log_s = self.add_weight(shape=(*[1 for _ in input_shape[:-1]], input_shape[-1]),
                                 initializer='zeros',
                                 name='log_s',
                                 dtype=tf.float32,
                                 trainable=True)
        self.b = self.add_weight(shape=(*[1 for _ in input_shape[:-1]], input_shape[-1]),
                                 initializer='zeros',
                                 name='b',
                                 dtype=tf.float32,
                                 trainable=True)

    def call(self, inputs):
        return tf.math.multiply(inputs + self.b, tf.math.exp(self.log_s))
