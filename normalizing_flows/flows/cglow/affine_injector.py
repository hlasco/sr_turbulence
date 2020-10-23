import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from flows.transform import Transform
from . import affine



def injector_nn_glow(dim=2, min_filters=32, max_filters=512, kernel_size=3, num_blocks=4, alpha=1.0E-5, epsilon=1.0E-4):
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

        h = Conv(dim, num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(alpha), name=f'{base_name}/conv{dim}d_2')(x)

        #h = ActNorm(name=f'{base_name}/act_norm_2')(h)
        h = Activation('relu')(h)

        for i in range(num_blocks):
            h = _resnet_block(h, dim, num_filters, f'{base_name}_{i}')

        s = Conv(dim, c_out, kernel_size, padding='same', kernel_regularizer=l2(alpha), kernel_initializer='zeros', name=f'{base_name}/conv{dim}d_s')(h)

        s = Activation('relu')(s)
        
        s = Lambda(lambda x: 2*1.9/np.pi * tf.math.atan(x/1.9), dtype=tf.float32)(s)
        s = Lambda(lambda x: log_scale + x, dtype=tf.float32)(s)
        s = Lambda(lambda x: tf.math.exp(x), dtype=tf.float32)(s)

        t = Conv(dim, c_out, kernel_size, padding='same', kernel_regularizer=l2(alpha), kernel_initializer='zeros', name=f'{base_name}/conv{dim}d_t')(h)
        model = Model(inputs=x, outputs=[s, t])
        return model
    return f

class AffineInjector(Transform):
    def __init__(self, layer, cond_shape, input_shape=None, nn_ctor=injector_nn_glow(),
                 name='affine_injector', *args, **kwargs):
        self.cond_shape = cond_shape
        self.layer = layer
        self.nn_ctor = nn_ctor
        self.nn = None
        super().__init__(*args, input_shape=input_shape, requires_init=True, name=name, **kwargs)

    def _initialize(self, input_shape):
        if self.nn is None:
            c_in  = self.cond_shape
            c_out = input_shape[-1]

            self.log_scale = tf.Variable(tf.zeros((1,1,1,c_out), dtype=tf.float32), dtype=tf.float32, name=f'{self.name}/log_scale')
            dim = input_shape.rank-2
            self.nn = self.nn_ctor(dim, self.layer, c_in, c_out, self.log_scale, self.name)

    def _forward(self, x, **kwargs):

        y_cond = kwargs['y_cond']
        s, t = self.nn(y_cond)
        y, fldj = affine(x, s, t)
        return y, fldj

    def _inverse(self, y, **kwargs):
        y_cond = kwargs['y_cond']
        s, t = self.nn(y_cond)
        x, ildj = affine(y, s, t, inverse=tf.constant(True))
        return x, ildj

    def _regularization_loss(self):
        assert self.nn is not None, 'bijector not initialized'
        return tf.math.reduce_sum(self.nn.get_losses_for(None))

    def _test(self, shape, **kwargs):
        print('Testing', self.name)
        normal = tfp.distributions.Normal(loc=0.0, scale=1.0)
        x = normal.sample(shape)

        normal = tfp.distributions.Normal(loc=0, scale=1)

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
