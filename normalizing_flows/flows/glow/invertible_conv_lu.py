import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import scipy
from flows.transform import Transform
from models.utils import var

def conv(x, W, padding='SAME'):
    if W.shape.rank ==4:
        y = tf.nn.conv2d(x, W, [1,1,1,1], padding='SAME')
    else:
        y = tf.nn.conv3d(x, W, [1,1,1,1,1], padding='SAME')
    return y

@tf.function
def compute_w(c, L, U, P, log_d, sgn_d, inverse=tf.constant(False)):
    tril_mask = tf.constant(np.tril(np.ones((1,c,c)), k=-1), dtype=tf.float32)
    d = tf.linalg.diag(tf.math.exp(log_d)*sgn_d)
    if inverse:
        L_inv = tf.linalg.inv(tril_mask*L + tf.eye(c, dtype=tf.float32))
        
        U_inv = tf.linalg.inv(tf.transpose(tril_mask, [0,2,1])*U + d)
        P_inv = tf.linalg.inv(P)
        W = tf.linalg.matmul(U_inv, tf.linalg.matmul(L_inv, P_inv))
    else:
        L = tril_mask*L + tf.eye(c, dtype=tf.float32)
        U = tf.transpose(tril_mask, [0,2,1])*U + d
        W = tf.linalg.matmul(P, tf.linalg.matmul(L, U))
    return tf.expand_dims(W, axis=0) # (1,1,c,c)

@tf.function
def invertible_1x1_conv(x, L, U, P, log_d, sgn_d, inverse=tf.constant(False)):
    shape = tf.shape(x)
    size = tf.cast(tf.math.reduce_prod(shape[1:-1]), tf.float32)
    W = compute_w(x.shape[-1], L, U, P, log_d, sgn_d, inverse=inverse)

    if x.shape.rank ==5:
        W = tf.expand_dims(W, axis=0)
    y = conv(x, W, padding='SAME')
    ldj = tf.math.reduce_sum(log_d)*tf.ones(shape[:1], tf.float32)*size
    if inverse:
        ldj *= -1
    return y, ldj

class InvertibleConv(Transform):
    def __init__(self, input_shape=None, alpha=1.0E-4, name='invertible_1x1_conv_lu', *args, **kwargs):
        self.alpha = alpha
        self.init = False
        super().__init__(*args,
                         input_shape=input_shape,
                         name=name,
                         requires_init=True,
                         **kwargs)

    def _initialize(self, input_shape):
        if not self.init:
            rank = input_shape.rank
            assert rank == 4 or rank==5, f'input should be 4 or 5-dimensional, got {input_shape}'
            c = input_shape[-1]
            # sample random orthogonal matrix and compute LU decomposition
            q,_ = np.linalg.qr(np.random.randn(c, c))
            p, l, u = scipy.linalg.lu(q)
            d = np.diag(u)
            # parameterize diagonal d as log(d) for numerical stability
            log_d = np.log(np.abs(d))
            sgn_d = np.sign(d)
            l, u = np.tril(l, k=-1), np.triu(u, k=1)
            # initialize variables
            self.input_shape = input_shape
            self.P = tf.Variable(np.expand_dims(p, axis=0).astype(np.float32), trainable=False, name=f'{self.name}/P')
            self.L = tf.Variable(np.expand_dims(l, axis=0).astype(np.float32), trainable=True, name=f'{self.name}/L')
            self.U = tf.Variable(np.expand_dims(u, axis=0).astype(np.float32), trainable=True, name=f'{self.name}/U')
            self.log_d = tf.Variable(np.expand_dims(log_d, axis=0).astype(np.float32), trainable=True, name=f'{self.name}/log_d')
            self.sgn_d = tf.Variable(np.expand_dims(sgn_d, axis=0).astype(np.float32), trainable=False, name=f'{self.name}/sgn_d')
            #self.tril_mask = tf.constant(np.tril(np.ones((1,c,c)), k=-1), dtype=tf.float32)
            self.init = True
    
    def _forward(self, x, **kwargs):
        return invertible_1x1_conv(x, var(self.L), var(self.U), var(self.P), var(self.log_d), var(self.sgn_d))
    
    def _inverse(self, y, **kwargs):
        return invertible_1x1_conv(y, var(self.L), var(self.U), var(self.P), var(self.log_d), var(self.sgn_d), inverse=tf.constant(True))
    
    def _regularization_loss(self):
        return self.alpha*tf.math.reduce_sum(self.log_d**2)

    def _test(self, shape, **kwargs):
        print('Testing', self.name)
        normal = tfp.distributions.Normal(loc=0.0, scale=1.0)
        x = normal.sample(shape)
        y, fldj = self._forward(x)
        x_, ildj = self._inverse(y)
        np.testing.assert_array_almost_equal(x_, x, decimal=2)
        np.testing.assert_array_equal(ildj, -fldj)
        err_x = tf.reduce_mean(x_-x)
        err_ldj = tf.reduce_mean(ildj+fldj)
        print("\tError on forward inverse pass:")
        print("\t\tx-F^{-1}oF(x):", err_x.numpy())
        print("\t\tildj+fldj:", err_ldj.numpy())
        print('\t passed')

    def param_count(self, _):
        return tf.size(self.P) + tf.size(self.L) + tf.size(self.U) + tf.size(self.log_d) + tf.size(self.sgn_d)
    