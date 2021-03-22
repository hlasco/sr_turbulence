import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from flows.transform import Transform
from flows.networks.coupling_nn import coupling_nn
from . import affine

class CondAffineCoupling(Transform):
    def __init__(self, layer, cond_shape, input_shape=None, nn_ctor=coupling_nn(),
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

            self.steep_s = tf.Variable(.5*tf.ones((1,1,1,c_out), dtype=tf.float32), dtype=tf.float32, name=f'{self.name}/steep_s')
            self.scale_s = tf.Variable(.5*tf.ones((1,1,1,c_out), dtype=tf.float32), dtype=tf.float32, name=f'{self.name}/scale_s')
            self.log_scale_t = tf.Variable(tf.zeros((1,1,1,c_out), dtype=tf.float32), dtype=tf.float32, name=f'{self.name}/log_scale_t')
            dim = input_shape.rank-2
            self.nn = self.nn_ctor(dim, self.layer, c_in, c_out, self.steep_s, self.scale_s, self.log_scale_t, self.name)

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
        return tf.size(self.scale_s) + tf.size(self.steep_s) + tf.size(self.log_scale_t) + self.nn.count_params()
