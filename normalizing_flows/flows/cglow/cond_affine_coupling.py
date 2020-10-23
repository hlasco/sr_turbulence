import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from flows.transform import Transform
from . import affine

def cond_coupling_nn_glow(dim=2, min_filters=32, max_filters=512, kernel_size=3, num_blocks=4, alpha=1.0E-5, epsilon=1.0E-4):
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Input, Conv2D, Conv3D,  BatchNormalization, Activation, Lambda, add
    from tensorflow.keras.regularizers import l2
    from flows.layers.act_norm import ActNorm
    def Conv(dim, num_filters, kernel_size, **kwargs):
        if dim==2:
            return Conv2D(num_filters, kernel_size, dtype=tf.float32, **kwargs)
        else:
            return Conv3D(num_filters, kernel_size, dtype=tf.float32, **kwargs)
    
    def _resnet_block(x, dim, num_filters, base_name):
        h = Conv(dim, num_filters, kernel_size, padding='same', kernel_regularizer=l2(alpha), name=f'{base_name}/conv{dim}d_1')(x)
        h = ActNorm(name=f'{base_name}/act_norm_1')(h)
        h = Activation('relu')(h)
        h = Conv(dim, num_filters, kernel_size, padding='same', kernel_regularizer=l2(alpha), name=f'{base_name}/conv{dim}d_2')(h)
        h = ActNorm(name=f'{base_name}/act_norm_2')(h)
        h = add([x, h])
        h = Activation('relu')(h)
        return h

    def f(dim, i, c_in, c_out, log_scale: tf.Variable, base_name):
        num_filters = np.maximum(max_filters//((2**(dim-1))**i), min_filters)
        if dim==2:
            x = Input((None,None,c_in), dtype=tf.float32)
        else:
            x = Input((None, None,None,c_in), dtype=tf.float32)
        h = Conv(dim, num_filters, kernel_size, padding='same', kernel_regularizer=l2(alpha), name=f'{base_name}/conv{dim}d_1')(x)
        
        #h = ActNorm(name=f'{base_name}/act_norm_1')(h)
        h = Activation('relu')(h)

        h = Conv(dim, num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(alpha), name=f'{base_name}/conv{dim}d_2')(h)

        #h = ActNorm(name=f'{base_name}/act_norm_2')(h)
        h = Activation('relu')(h)

        for i in range(num_blocks):
            h = _resnet_block(h, dim, num_filters, f'{base_name}_{i}')

        s = Conv(dim, c_out, kernel_size, padding='same', kernel_regularizer=l2(alpha), kernel_initializer='zeros', name=f'{base_name}/conv{dim}d_s')(h)

        s = Activation('relu')(s)
        
        s = Lambda(lambda x: 2*1.9/np.pi * tf.math.atan(x/1.9), dtype=tf.float32)(s)
        s = Lambda(lambda x: log_scale + x, dtype=tf.float32)(s)
        s = Lambda(lambda x: tf.math.exp(x), dtype=tf.float32)(s)

        t = Conv(dim, c_out, kernel_size, padding='same', kernel_regularizer=l2(alpha),  kernel_initializer='zeros', name=f'{base_name}/conv{dim}d_t')(h)
        model = Model(inputs=x, outputs=[s, t])
        return model
    return f

class CondAffineCoupling(Transform):
    def __init__(self, layer, cond_shape, input_shape=None, nn_ctor=cond_coupling_nn_glow(),
                 reverse=False, name='cond_affine_coupling', *args, **kwargs):
        self.cond_shape = cond_shape
        self.layer = layer
        self.nn_ctor = nn_ctor
        self.reverse = reverse
        self.nn = None
        super().__init__(*args, input_shape=input_shape, requires_init=True, name=name, **kwargs)

    def _initialize(self, input_shape):
        if self.nn is None:
            c_in  = input_shape[-1]//2 + self.cond_shape
            c_out = input_shape[-1]//2

            self.log_scale = tf.Variable(tf.zeros((1,1,1,c_out), dtype=tf.float32), dtype=tf.float32, name=f'{self.name}/log_scale')
            dim = input_shape.rank-2
            self.nn = self.nn_ctor(dim, self.layer, c_in, c_out, self.log_scale, self.name)

    def _forward(self, x, **kwargs):
        if self.reverse:
            x = x[...,::-1]
        x_a, x_b = tf.split(x, 2, axis=-1)
        y_cond = kwargs['y_cond']

        x_b_cond = tf.concat([x_b, y_cond], axis=-1)
        s, t = self.nn(x_b_cond)
        y_a, fldj = affine(x_a, s, t)
        y_b = x_b
        y = tf.concat([y_a, y_b], axis=-1)

        return y, fldj

    def _inverse(self, y, **kwargs):

        y_a, y_b = tf.split(y, 2, axis=-1)
        y_cond = kwargs['y_cond']

        y_b_cond = tf.concat([y_b, y_cond], axis=-1)
        s, t = self.nn(y_b_cond)
        x_a, ildj = affine(y_a, s, t, inverse=tf.constant(True))
        x_b = y_b
        x = tf.concat([x_a, x_b], axis=-1)

        if self.reverse:
            x = x[...,::-1]
        return x, ildj

    def _regularization_loss(self):
        assert self.nn is not None, 'bijector not initialized'
        return tf.math.reduce_sum(self.nn.get_losses_for(None))

    def _test(self, shape, **kwargs):
        print('Testing', self.name)
        normal = tfp.distributions.Normal(loc=0.1, scale=1.0)
        x = normal.sample(shape)

        y_shape = kwargs['y_shape']
        cond_shape = tf.concat([shape[:-1], [self.cond_shape]], axis=-1)
        y_cond =  normal.sample(cond_shape)
        y_cond = tf.cast(y_cond, tf.float32)
        y, fldj = self._forward(x, y_cond=y_cond)
        x_, ildj = self._inverse(y, y_cond=y_cond)
        np.testing.assert_array_almost_equal(x_, x, decimal=5)
        np.testing.assert_array_equal(ildj, -fldj)
        err_x = tf.reduce_mean(x_-x)
        err_ldj = tf.reduce_mean(ildj+fldj)
        print("\tError on forward inverse pass:")
        print("\t\tx-F^{-1}oF(x):", err_x.numpy())
        print("\t\tildj+fldj:", err_ldj.numpy())
        print('\t passed')

    def param_count(self, shape):
        return tf.size(self.log_scale) + self.nn.count_params()
