import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
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

def Gaussianize(dim=2, min_filters=32, max_filters=32, *args, **kwargs):
    class _gaussianize(Parameterize):
        """
        Implementation of parameterize for a Gaussian prior. Corresponds to the "Gaussianization" step in Glow (Kingma et al, 2018).
        """
        def __init__(self, i=0, input_shape=None, name='gaussianize', *args, **kwargs):
            self.num_filters = np.maximum(max_filters//((2**(dim-1))**i), min_filters)
            if 'cond_channels' in kwargs:
                kwargs['cond_channels'] = 0
            super().__init__(*args, num_parameters=2, num_filters=self.num_filters, input_shape=input_shape, name=name, **kwargs)
            
        def _forward(self, x1, x2, **kwargs):
            params = self.parameterizer(x1)
            mus, log_sigmas = params[...,0::2], params[...,1::2]
            z2, fldj = gaussianize(x2, mus, log_sigmas)
            return z2, fldj
        
        def _inverse(self, x1, z2, **kwargs):
            params = self.parameterizer(x1)
            mus, log_sigmas = params[...,0::2], params[...,1::2]
            x2, ildj = gaussianize(z2, mus, log_sigmas, inverse=tf.constant(True))
            return x2, ildj
    
        def _test(self, shape, **kwargs):
            import numpy as np
            print('Testing', self.name)
            x = tf.random.normal(shape, dtype=tf.float32)
            z, fldj = self._forward(tf.zeros_like(x), x, dtype=tf.float32)
            x_, ildj = self._inverse(tf.zeros_like(x), z, dtype=tf.float32)
            #np.testing.assert_array_almost_equal(np.mean(z), 0.0, decimal=2)
            #np.testing.assert_array_almost_equal(np.std(z), 1.0, decimal=2)
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
    return _gaussianize
    
def log_gaussianize(x, mus, log_sigmas, inverse=tf.constant(False)):
    """
    Standardize log normal random variable x using mus and log_sigmas.
    """
    if inverse:
        scales = tf.math.exp(log_sigmas)
        log_x = tf.math.log(x)
        ldj = log_x
        log_y = log_x*scales + mus
        ldj += log_sigmas
        z = tf.math.exp(log_y)
        return z, ldj
    else:
        scales = tf.math.exp(-log_sigmas)
        log_x = tf.math.log(x)
        ldj = -log_x
        log_y = (log_x - mus)*scales
        ldj -= log_sigmas
        z = tf.math.exp(log_y)
        return z, ldj

class LogGaussianize(Parameterize):
    """
    Implementation of Parameterize for a log-Gaussian prior.
    """
    def __init__(self, input_shape=None, epsilon=1.0E-3, name='log_gaussianize', *args, **kwargs):
        super().__init__(*args, num_parameters=2, input_shape=input_shape, name=name, **kwargs)
        self.epsilon = epsilon
        
    def _forward(self, x1, x2, **kwargs):
        """
        A log normal RV X = exp(mu + sigma*Z) where Z ~ N(0,I).
        The forward pass scales to a standard log normal with mu=0, sigma=1 by computing:
        exp(Z) = (X / exp(mu))^(1/sigma)
        """
        rank = tf.shape(x2).rank
        params = self.parameterizer(x1)
        mus, log_sigmas = params[...,0::2], params[...,1::2]
        # compute softplus activation
        z2, ldj = log_gaussianize(x2, mus, log_sigmas)
        z2 = tf.where(x2 > self.epsilon, z2, x2)
        ldj = tf.where(x2 > self.epsilon, ldj, tf.zeros_like(ldj))
        return z2, tf.math.reduce_sum(ldj, axis=[[i for i in range(1, rank)]])
    
    def _inverse(self, x1, z2, **kwargs):
        params = self.parameterizer(x1)
        mus, log_sigmas = params[...,0::2], params[...,1::2]
        x2, ldj = log_gaussianize(z2, mus, log_sigmas, inverse=tf.constant(True))
        x2 = tf.where(z2 > self.epsilon, x2, z2)
        ldj = tf.where(z2 > self.epsilon, ldj, tf.zeros_like(ldj))
        return x2, tf.math.reduce_sum(ldj, axis=[[i for i in range(1, rank)]])
    
def half_gaussianize(x, log_sigmas, inverse=tf.constant(False)):
    rank = tf.shape(x).rank
    if inverse:
        z = tf.math.exp(log_sigmas)*x
        ldj = tf.math.reduce_sum(log_sigmas, axis=[[i for i in range(1, rank)]])
    else:
        z = x*tf.math.exp(-log_sigmas)
        ldj = -tf.math.reduce_sum(log_sigmas, axis=[[i for i in range(1, rank)]])
    return z, ldj

class HalfGaussianize(Parameterize):
    """
    Implementation of parameterize for a half-Gaussian prior.
    """
    def __init__(self, input_shape=None, name='gaussianize', *args, **kwargs):
        super().__init__(*args, num_parameters=1, input_shape=input_shape, name=name, **kwargs)
        
    def _forward(self, x1, x2, **kwargs):
        log_sigmas = self.parameterizer(x1)
        z2, fldj = half_gaussianize(x2, log_sigmas)
        return z2, fldj
    
    def _inverse(self, x1, z2, **kwargs):
        log_sigmas = self.parameterizer(x1)
        x2, ildj = half_gaussianize(z2, log_sigmas, inverse=tf.constant(True))
        return x2, ildj
    
def exponentiate(x, log_lambdas, inverse=tf.constant(False)):
    rank = tf.shape(x).rank
    if not inverse:
        z = tf.math.exp(log_lambdas)*x
        ldj = tf.math.reduce_sum(log_lambdas, axis=[[i for i in range(1, rank)]])
    else:
        z = x*tf.math.exp(-log_lambdas)
        ldj = -tf.math.reduce_sum(log_lambdas, axis=[[i for i in range(1, rank)]])
    return z, ldj

class Exponentiate(Parameterize):
    """
    Implementation of parameterize for an exponetial prior.
    """
    def __init__(self, input_shape=None, name='gaussianize', *args, **kwargs):
        super().__init__(*args, num_parameters=1, input_shape=input_shape, name=name, **kwargs)
        
    def _forward(self, x1, x2, **kwargs):
        log_lambdas = self.parameterizer(x1)
        z2, fldj = exponentiate(x2, log_lambdas)
        return z2, fldj
    
    def _inverse(self, x1, z2, **kwargs):
        log_lambdas = self.parameterizer(x1)
        x2, ildj = exponentiate(z2, log_lambdas, inverse=tf.constant(True))
        return x2, ildj
