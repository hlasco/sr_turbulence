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
                self.W = ortho_init((1,1,c,c))
            else:
                self.W = ortho_init((1,1,1,c,c))
            #self.W = tf.reshape(tf.eye(c,c), (1,1,c,c))
    
    def _forward(self, x):
        self._initialize(tf.shape(x).shape)
        y = conv(x, self.W, padding='SAME')
        fldj = tf.math.log(tf.math.abs(tf.linalg.det(self.W)))
        fldj = tf.squeeze(fldj)
        return y, tf.broadcast_to(fldj, (tf.shape(x)[0],))
    
    def _inverse(self, y):
        #self._init_vars(y)
        self._initialize(tf.shape(y).shape)
        W_inv = tf.linalg.inv(self.W)
        x = conv(y, W_inv, padding='SAME')
        ildj = tf.math.log(tf.math.abs(tf.linalg.det(W_inv)))
        ildj = tf.squeeze(ildj)
        return x, tf.broadcast_to(ildj, (y.shape[0],))

    def param_count(self, _):
        return tf.size(self.W)
    