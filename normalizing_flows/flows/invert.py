import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from .transform import Transform

class Invert(Transform):
    def __init__(self, transform, *args, **kwargs):
        self.transform = transform
        super().__init__(*args,
                         input_shape=transform.input_shape,
                         has_constant_ldj=transform.has_constant_ldj,
                         name=f'inverse_{transform.name}', **kwargs)
    
    def _initialize(self, input_shape):
        self.transform.initialize(input_shape)

    def _forward(self, z, *args, **kwargs):
        return self.transform.inverse(z, *args, **kwargs)

    def _inverse(self, z, *args, **kwargs):
        return self.transform.forward(z, *args, **kwargs)
    
    def _forward_shape(self, shape: tf.TensorShape):
        return self.transform._inverse_shape(shape)
    
    def _inverse_shape(self, shape: tf.TensorShape):
        return self.transform._forward_shape(shape)
        
    def _regularization_loss(self):
         # note: this will double count regularization if transform is added separately to the flow
        return self.transform.regularization_loss()

    def _test(self, shape, **kwargs):
        return self.transform._test(shape, **kwargs)

    def _cond_fn(self):
        return self.transform.cond_fn

    def _get_trainable_variables(self, **kwargs):
        return self.transform.get_trainable_variables( **kwargs)

    def param_count(self, shape=None):
        # note: this will double count parameters if transform is added separately to the flow
        return self.transform.param_count(shape)