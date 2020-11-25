import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from flows.transform import Transform
from flows.networks.coupling_nn import coupling_nn

def affine(x, s, t, inverse=tf.constant(False)):
    rank = x.shape.rank
    ldj =  tf.math.reduce_sum(tf.math.log(s), axis=[i for i in range(1,rank)])
    if inverse:
        y = x / s - t
        ldj *= -1
    else:
        y = (x + t)*s
    return y, ldj

class AffineCoupling(Transform):
    def __init__(self, layer, input_shape=None, nn_ctor=coupling_nn(), name='affine_coupling', *args, **kwargs):
        self.layer = layer
        self.nn_ctor = nn_ctor
        self.nn = None
        super().__init__(*args, input_shape=input_shape, requires_init=True, name=name, **kwargs)

    def _initialize(self, input_shape):
        if self.nn is None:
            c_in  = input_shape[-1]//2
            c_out = input_shape[-1]//2
            self.log_scale_s = tf.Variable(tf.zeros((1,1,1,c_out), dtype=tf.float32), dtype=tf.float32, name=f'{self.name}/log_scale_s')
            self.log_scale_t = tf.Variable(tf.zeros((1,1,1,c_out), dtype=tf.float32), dtype=tf.float32, name=f'{self.name}/log_scale_t')
            dim = input_shape.rank - 2
            self.nn = self.nn_ctor(dim, self.layer, c_in, c_out, self.log_scale_s, self.log_scale_t, self.name)

    def _forward(self, x, **kwargs):
        x_a, x_b = tf.split(x, 2, axis=-1)
        s, t = self.nn(x_b)
        y_a, fldj = affine(x_a, s, t)
        y_b = x_b
        return tf.concat([y_a, y_b], axis=-1), fldj

    def _inverse(self, y, **kwargs):
        y_a, y_b = tf.split(y, 2, axis=-1)
        s, t = self.nn(y_b)
        x_a, ildj = affine(y_a, s, t, inverse=tf.constant(True))
        x_b = y_b
        return tf.concat([x_a, x_b], axis=-1), ildj

    def _regularization_loss(self):
        assert self.nn is not None, 'bijector not initialized'
        return tf.math.reduce_sum(self.nn.get_losses_for(None))

    def _test(self, shape, **kwargs):
        print('Testing', self.name)
        normal = tfp.distributions.Normal(loc=0.0, scale=1.0)
        x = normal.sample(shape)
        y, fldj = self._forward(x, **kwargs)
        x_, ildj = self._inverse(y, **kwargs)
        np.testing.assert_array_almost_equal(x_, x, decimal=5)
        np.testing.assert_array_equal(ildj, -fldj)
        err_x = tf.reduce_mean(x_-x)
        err_ldj = tf.reduce_mean(ildj+fldj)
        print("\tError on forward inverse pass:")
        print("\t\tx-F^{-1}oF(x):", err_x.numpy())
        print("\t\tildj+fldj:", err_ldj.numpy())
        print('\t passed')

    def param_count(self, shape):
        return tf.size(self.log_scale_s) + tf.size(self.log_scale_t) + self.nn.count_params()
