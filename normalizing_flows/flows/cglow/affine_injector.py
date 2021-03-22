import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from flows.transform import Transform
from flows.networks.coupling_nn import coupling_nn
from . import affine

class AffineInjector(Transform):
    def __init__(self, layer, cond_shape, input_shape=None, nn_ctor=coupling_nn(),
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

            self.steep_s = tf.Variable(.5*tf.ones((1,1,1,c_out), dtype=tf.float32), dtype=tf.float32, name=f'{self.name}/steep_s')
            self.scale_s = tf.Variable(.5*tf.ones((1,1,1,c_out), dtype=tf.float32), dtype=tf.float32, name=f'{self.name}/scale_s')
            self.log_scale_t = tf.Variable(tf.zeros((1,1,1,c_out), dtype=tf.float32), dtype=tf.float32, name=f'{self.name}/log_scale_t')
            dim = input_shape.rank-2
            self.nn = self.nn_ctor(dim, self.layer, c_in, c_out, self.steep_s, self.scale_s, self.log_scale_t, self.name)

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
        print("\t\tx-F^{-1}oF(x):", err_x)
        print("\t\tildj+fldj:", err_ldj)
        print('\t passed')

    def param_count(self, shape):
        return tf.size(self.scale_s) + tf.size(self.steep_s) + tf.size(self.log_scale_t) + self.nn.count_params()
