import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from flows.transform import Transform
from . import Parameterize


def gaussianize(x, mus, log_sigmas, inverse=tf.constant(False)):
    rank = x.shape.rank
    if inverse:
        z = tf.math.exp(log_sigmas)*x + mus
        ldj = tf.math.reduce_sum(log_sigmas, axis=[i for i in range(1, rank)])
    else:
        z = (x - mus)*tf.math.exp(-log_sigmas)
        ldj = -tf.math.reduce_sum(log_sigmas, axis=[i for i in range(1, rank)])
    return z, ldj

def CondGaussianize(dim=2, min_filters=32, max_filters=32, *args, **kwargs):
    class _cond_gaussianize(Parameterize):
        """
        Implementation of parameterize for a Gaussian prior. Corresponds to the "Gaussianization" step in Glow (Kingma et al, 2018),
        with a conditional input.
        """
    
        def __init__(self, i=0, cond_channels=0, input_shape=None, name='cond_gaussianize', *args, **kwargs):
            self.cond_channels = cond_channels
            self.num_filters = np.maximum(max_filters//((2**(dim-1))**i), min_filters)
            super().__init__(*args, num_parameters=2, num_filters=self.num_filters, input_shape=input_shape, cond_channels=cond_channels, name=name, **kwargs)
            
        def _forward(self, x1, x2, **kwargs):
            assert 'y_cond' in kwargs, 'cond_gaussianize did not receive y_cond'
            y_cond = kwargs['y_cond']
            x1 = tf.concat([x1, y_cond], axis=-1)
            params = self.parameterizer(x1)
            mus, log_sigmas = params[...,0::2], params[...,1::2]
            z2, fldj = gaussianize(x2, mus, log_sigmas)
            return z2, fldj
        
        def _inverse(self, x1, z2, **kwargs):
            assert 'y_cond' in kwargs, 'cond_gaussianize did not receive y_cond'
            y_cond = kwargs['y_cond']
            x1 = tf.concat([x1, y_cond], axis=-1)
            params = self.parameterizer(x1)
            mus, log_sigmas = params[...,0::2], params[...,1::2]
            x2, ildj = gaussianize(z2, mus, log_sigmas, inverse=tf.constant(True))
            return x2, ildj
    
        def _test(self, shape, **kwargs):
            import numpy as np
            print('Testing', self.name)
            x = tf.random.normal(shape, dtype=tf.float32)
            normal = tfp.distributions.Normal(loc=0, scale=1)
    
            cond_shape = tf.concat([shape[:-1], [self.cond_channels]], axis=-1)
            y_cond = normal.sample(cond_shape)
            z, fldj = self._forward(tf.zeros_like(x), x, y_cond=y_cond)
            x_, ildj = self._inverse(tf.zeros_like(x), z, y_cond=y_cond)
            np.testing.assert_array_almost_equal(x_, x, decimal=5)
            np.testing.assert_array_equal(ildj, -fldj)
            err_x = tf.reduce_mean(x_-x)
            err_ldj = tf.reduce_mean(ildj+fldj)
            print("\tError on forward inverse pass:")
            print("\t\tx-F^{-1}oF(x):", err_x.numpy())
            print("\t\tildj+fldj:", err_ldj.numpy())
            print('\t passed')
    
        def param_count(self, _):
            return self.parameterizer.count_params()

    return _cond_gaussianize
        