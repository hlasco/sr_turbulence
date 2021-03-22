import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from flows.transform import Transform
from tensorflow.keras import backend as K

def fwd_s(shape, factor):
    ret = [s//factor if s is not None else None for s in shape]
    return ret

def inv_s(shape, factor):
    ret = [s*factor if s is not None else None for s in shape]
    return ret

class Squeeze(Transform):
    def __init__(self, input_shape=None, factor=2, *args, **kwargs):
        """
        Creates a new transform for the "squeeze" operation, where spatial dimensions are folded
        into channel dimensions. This bijector requires the input data to be 3-dimensional,
        height-width-channel (HWC) or 4-dimensional, height-width-depth-channel (HWC)
        formatted images/boxes (exluding the batch axis).
        """
        self.factor = factor
        super().__init__(*args,
                         input_shape=input_shape,
                         requires_init=False,
                         has_constant_jacobian=True,
                         **kwargs)
        
    def _initialize(self, shape):
        if shape.rank == 4:
            self.transpose_fwd = [0, 1, 3, 5, 2, 4]
            self.transpose_inv = [0, 1, 4, 2, 5, 3]
        elif shape.rank == 5:
            self.transpose_fwd = [0, 1, 3, 5, 7, 2, 4, 6]
            self.transpose_inv = [0, 1, 5, 2, 6, 3, 7, 4]
        self.dim = shape.rank-2

    def _forward(self, x, *args, **kwargs):
        shape = x.shape
        factor = self.factor
        c = shape[-1]

        shape_1 = fwd_s(shape[1:-1], factor)
        shape_2 = fwd_s(shape[1:-1], factor)

        for b in range (0,len(shape_1)):
            shape_1.insert(b*2+1,factor)
        # reshape to intermediate tensor
        x_ = tf.reshape(x, (-1, *shape_1, c))
        # transpose factored out dimensions to channel axis
        x_ = tf.transpose(x_, self.transpose_fwd)
        # reshape to final output shape
        y = tf.reshape(x_, (-1, *shape_2, c*factor**self.dim))
        return y, 0.0

    def _inverse(self, y, *args, **kwargs):
        shape = y.shape
        factor = self.factor
        c = shape[-1]
        c_factored = c // (factor**self.dim)
        # reshape to intermediate tensor
        y_ = tf.reshape(y, (-1, *shape[1:-1], c_factored, *[factor for _ in range(self.dim)]))
        # transpose factored out dimensions back to original intermediate axes
        y_ = tf.transpose(y_, self.transpose_inv)
        # reshape to final output shape
        #s1, s2 = inv_s(shape[1], factor), inv_s(shape[2], factor)
        shape_1 = inv_s(shape[1:-1], factor)
        x = tf.reshape(y_, (-1, *shape_1, c_factored))
        return x, 0.0
    
    def _forward_shape(self, shape):
        assert self.input_shape is not None, 'not initialized'
        factor = self.factor
        #s1, s2, s3 = fwd_s(shape[1], factor), fwd_s(shape[2], factor), fwd_s(shape[3], factor)
        shape_1 = fwd_s(shape[1:-1], factor)
        return tf.TensorShape((shape[0], *shape_1, int(shape[-1]*factor**self.dim)))
    
    def _inverse_shape(self, shape):
        assert self.input_shape is not None, 'not initialized'
        factor = self.factor
        shape_1 = inv_s(shape[1:-1], factor)
        return tf.TensorShape((shape[0], *shape_1, int(shape[-1]//(factor**self.dim))))

    def _test(self, shape, **kwargs):
        print('Testing', self.name)
        normal = tfp.distributions.Normal(loc=0.0, scale=1.0)
        x = normal.sample(shape)
        y,_ = self._forward(x, **kwargs)
        fwd_shape = self._forward_shape(shape)
        np.testing.assert_array_equal(tf.shape(y), fwd_shape)
        x_,_ = self._inverse(y, **kwargs)
        np.testing.assert_array_equal(tf.shape(x_), x.shape)
        
        print('\t passed')

    def param_count(self, _):
        return 0
