import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from flows.transform import Transform
from tensorflow.keras import backend as K

def fwd_s(shape, factor):
    if shape==None:
        return None
    return shape // factor

def inv_s(shape, factor):
    if shape==None:
        return None
    return shape * factor

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
        
    #def _initialize(self, shape):
    #    if self.padding_x is None or self.padding_y is None:
    #        assert shape.rank == 4, f'input should be 4-dimensional, got {shape}'
    #        batch_size, ht, wt, c = shape[0], shape[1], shape[2], shape[3]
    #        self.padding_y, self.padding_x = 0,0#ht % self.factor, wt % self.factor

    def _forward2d(self, x, *args, **kwargs):
        shape = x.shape
        factor = self.factor
        h, w, c = shape[1:]
        # pad to divisor
        assert h is not None and w is not None, 'height and width must be known'
        s1, s2 = fwd_s(shape[1], factor), fwd_s(shape[2], factor)
        # reshape to intermediate tensor
        x_ = tf.reshape(x, (-1, s1, factor, s2, factor, c))
        # transpose factored out dimensions to channel axis
        x_ = tf.transpose(x_, [0, 1, 3, 5, 2, 4])
        # reshape to final output shape
        y = tf.reshape(x_, (-1, s1, s2, c*factor**2))
        return y, 0.0

    def _inverse2d(self, y, *args, **kwargs):
        shape = y.shape
        factor = self.factor
        h, w, c = shape[1:]
        c_factored = c // (factor**2)
        # reshape to intermediate tensor
        y_ = tf.reshape(y, (-1, h, w, c_factored, factor, factor))
        # transpose factored out dimensions back to original intermediate axes
        y_ = tf.transpose(y_, [0, 1, 4, 2, 5, 3])
        # reshape to final output shape
        s1, s2 = inv_s(shape[1], factor), inv_s(shape[2], factor)
        x = tf.reshape(y_, (-1, s1, s2, c_factored))
        return x, 0.0
    
    def _forward_shape2d(self, shape):
        assert self.input_shape is not None, 'not initialized'
        factor = self.factor
        s1, s2 = fwd_s(shape[1], factor), fwd_s(shape[2], factor)
        c_factored = shape[3] * (factor**2)
        ret = tf.TensorShape((shape[0], s1, s2, c_factored))
        #ret = [shape[0], int(s1), int(s2), c_factored]
        #print(ret)
        return ret
    
    def _inverse_shape2d(self, shape):
        assert self.input_shape is not None, 'not initialized'
        factor = self.factor
        s1, s2 = inv_s(shape[1], factor), inv_s(shape[2], factor)
        c_factored = shape[3] // (factor**2)
        return tf.TensorShape((shape[0], s1, s2, c_factored))

    def _forward3d(self, x, *args, **kwargs):
        shape = x.shape
        factor = self.factor
        h, w, d, c = shape[1:]
        # pad to divisor
        assert h is not None and w is not None and d is not None, 'height, width and depth must be known'
        # reshape to intermediate tensor
        s1, s2, s3 = fwd_s(h, factor), fwd_s(w, factor), fwd_s(d, factor)
        x_ = tf.reshape(x, (-1, s1, factor, s2, factor, s3, factor, c))
        # transpose factored out dimensions to channel axis
        x_ = tf.transpose(x_, [0, 1, 3, 5, 7, 2, 4, 6])
        # reshape to final output shape
        y = tf.reshape(x_, (-1, s1, s2, s3, c*factor**3))
        return y, 0.0

    def _inverse3d(self, y, *args, **kwargs):
        shape = y.shape
        factor = self.factor
        h, w, d, c = shape[1:]
        c_factored = int(c // (factor**3))
        # reshape to intermediate tensor
        y_ = tf.reshape(y, (-1, h, w, d, c_factored, factor, factor, factor))
        # transpose factored out dimensions back to original intermediate axes
        y_ = tf.transpose(y_, [0, 1, 5, 2, 6, 3, 7, 4])
        # reshape to final output shape
        s1, s2, s3 = inv_s(h, factor), inv_s(w, factor), inv_s(d, factor)
        x = tf.reshape(y_, (-1, s1, s2, s3, c_factored))
        return x, 0.0
    
    def _forward_shape3d(self, shape):
        assert self.input_shape is not None, 'not initialized'
        factor = self.factor
        s1, s2, s3 = fwd_s(shape[1], factor), fwd_s(shape[2], factor), fwd_s(shape[3], factor)
        return tf.TensorShape((shape[0], s1, s2, s3, int(shape[4]*factor**3)))
    
    def _inverse_shape3d(self, shape):
        assert self.input_shape is not None, 'not initialized'
        factor = self.factor
        s1, s2, s3 = inv_s(shape[1], factor), inv_s(shape[2], factor), inv_s(shape[3], factor)
        return tf.TensorShape((shape[0], s1, s2, s3, int(shape[4]//(factor**3))))

    def _forward(self, x, *args, **kwargs):
        shape = x.shape
        if len(shape)==4:
            return self._forward2d(x, shape)
        else:
            return self._forward3d(x, shape)

    def _inverse(self, y, *args, **kwargs):
        shape = y.shape
        if len(shape)==4:
            return self._inverse2d(y, shape)
        else:
            return self._inverse3d(y, shape)

    def _forward_shape(self, shape):
        if len(shape)==4:
            return self._forward_shape2d(shape)
        else:
            return self._forward_shape3d(shape)

    def _inverse_shape(self, shape):
        if len(shape)==4:
            return self._inverse_shape2d(shape)
        else:
            return self._inverse_shape3d(shape)

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
