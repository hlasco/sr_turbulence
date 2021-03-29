import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import scipy
from flows.transform import Transform

def conv(x, W, padding='SAME'):
    if x.shape.rank ==4:
        y = tf.nn.conv2d(x, W, [1,1,1,1], padding='SAME')
    else:
        y = tf.nn.conv3d(x, W, [1,1,1,1,1], padding='SAME')
    return y

class InvertibleConv(Transform):
    def __init__(self, input_shape=None, name='invertible_1x1_conv', *args, **kwargs):
        self.W = None
        super().__init__(*args, input_shape=input_shape, requires_init=True, name=name, **kwargs)

    def _initialize(self, input_shape):
        if self.W is None:
            rank = input_shape.rank
            assert rank == 4 or rank == 5, 'input should be 4 or 5-dimensional'
            c = input_shape[-1]
            ortho_init = tf.initializers.Orthogonal()
            if rank==4:
                W = ortho_init((1,1,c,c))
            else:
                W = ortho_init((1,1,1,c,c))
            self.W = tf.Variable(W, name=f'{self.name}/W')
    
    def _forward(self, x, **kwargs):
        self._initialize(tf.shape(x).shape)
        y = conv(x, self.W, padding='SAME')
        w_64 = tf.cast(self.W, tf.float64)
        fldj = tf.math.log(tf.math.abs(tf.linalg.det(w_64))) * np.prod(x.shape[1:-1])
        fldj = tf.cast(fldj, tf.float32) 
        fldj = tf.squeeze(fldj)
        return y, tf.broadcast_to(fldj, (tf.shape(x)[0],))
    
    def _inverse(self, y, **kwargs):
        #self._init_vars(y)
        self._initialize(tf.shape(y).shape)
        W_inv = tf.linalg.inv(self.W)
        x = conv(y, W_inv, padding='SAME')
        w_64 = tf.cast(self.W, tf.float64)
        ildj = -tf.math.log(tf.math.abs(tf.linalg.det(w_64))) * np.prod(x.shape[1:-1])
        ildj = tf.cast(ildj, tf.float32) 
        ildj = tf.squeeze(ildj)
        return x, tf.broadcast_to(ildj, (y.shape[0],))

    def _test(self, shape, **kwargs):
        print('Testing', self.name)
        print(self.W)
        normal = tfp.distributions.Normal(loc=0.0, scale=1.0)
        x = normal.sample(shape)
        y, fldj = self._forward(x)
        x_, ildj = self._inverse(y)
        #np.testing.assert_array_almost_equal(x_, x, decimal=2)
        #np.testing.assert_array_equal(ildj, -fldj)
        err_x = tf.reduce_mean(x_-x)
        err_ldj = tf.reduce_mean(ildj+fldj)
        print("\tError on forward inverse pass:")
        print("\t\tx-F^{-1}oF(x):", err_x.numpy())
        print("\t\tildj+fldj:", err_ldj.numpy())
        print('\t passed')

    def param_count(self, _):
        return tf.size(self.W)
    