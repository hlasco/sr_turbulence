import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Conv3D,  BatchNormalization, Activation, Lambda, add, Concatenate
from tensorflow.keras.regularizers import l2

from . import ActNorm

def coupling_nn(dim=2, min_filters=32, max_filters=512, kernel_size=3, num_blocks=0, alpha=1.0E-5, epsilon=1.0E-4):
    def Conv(dim, num_filters, kernel_size, **kwargs):
        if dim==2:
            return Conv2D(num_filters, kernel_size, dtype=tf.float32, **kwargs)
        else:
            return Conv3D(num_filters, kernel_size, dtype=tf.float32, **kwargs)

    #def _resnet_block(x, dim, num_filters, base_name):
    #    h = Conv(dim, num_filters, kernel_size, padding='same', name=f'{base_name}/conv{dim}d_1')(x)
    #    #h = ActNorm(name=f'{base_name}/act_norm_1')(h)
    #    h = Activation('relu')(h)
    #    h = Conv(dim, num_filters, kernel_size, padding='same', name=f'{base_name}/conv{dim}d_2')(h)
    #    #h = ActNorm(name=f'{base_name}/act_norm_2')(h)
    #    h = add([x, h])
    #    h = Activation('relu')(h)
    #    return h

    def _resnet_block(x, dim, num_filters, base_name):
        w_init = tf.random_normal_initializer(stddev=0.02)
        u = x
        h = x
        for i in range(3):
            u = Conv(dim, num_filters, kernel_size, padding='same', kernel_initializer=w_init, name=f'{base_name}/conv{dim}d_{i}')(h)
            u = Activation('relu')(u)
            if i<5-1:
                h = Concatenate(axis=-1)([h,u])
        output = add([u, .2*x])
        return output

    def f(dim, i, c_in, c_out, steep_s: tf.Variable, scale_s: tf.Variable, log_scale_t: tf.Variable, base_name, eps=0.0001):
        #num_filters = np.maximum(max_filters//((2**(dim-1))**i), min_filters)
        num_filters = np.minimum(min_filters*((2**(dim-1))**i), max_filters)
        #print(i, num_filters)

        shape = (*[None for _ in range(dim)], c_in)
        x = Input(shape, dtype=tf.float32)
        h = Conv(dim, num_filters, kernel_size, padding='same', name=f'{base_name}/conv{dim}d_1')(x)
        h = Activation('relu')(h)
        h = Conv(dim, num_filters, kernel_size, padding='same', name=f'{base_name}/conv{dim}d_2')(h)
        h = Activation('relu')(h)

        for i in range(num_blocks):
            h = _resnet_block(h, dim, num_filters, f'{base_name}_resnet{i}')

        s = Conv(dim, c_out, kernel_size, padding='same', kernel_initializer='zeros', name=f'{base_name}/conv{dim}d_s')(h)

        s = Lambda(lambda x: steep_s*x, dtype=tf.float32)(s)
        s = Lambda(lambda x: scale_s*(tf.math.sigmoid(x)-.5), dtype=tf.float32)(s)
        s = Lambda(lambda x: tf.math.exp(x), dtype=tf.float32)(s)

        t = Conv(dim, c_out, kernel_size, padding='same',  kernel_initializer='zeros', name=f'{base_name}/conv{dim}d_t')(h)
        t = Lambda(lambda x: tf.math.exp(log_scale_t)*x, dtype=tf.float32)(t)
        model = Model(inputs=x, outputs=[s, t])
        return model
    return f
