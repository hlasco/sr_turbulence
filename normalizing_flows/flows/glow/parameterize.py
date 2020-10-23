import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Conv3D
from tensorflow.keras.initializers import RandomNormal
from flows.transform import Transform

class Parameterize(Transform):
    """
    Generalized base type for parameterizing a pre-specified density given some factored out latent variables.
    """
    def __init__(self, num_parameters, num_filters, input_shape=None, cond_shape=0, name='parameterize', *args, **kwargs):
        """
        Base class constructor. Should not be directly invoked by callers.

        num_parameters : number of distribution parameters per channel dimension (e.g. 2 for a Gaussian, mu and sigma)
        """
        self.num_parameters = num_parameters
        self.num_filters = num_filters
        self.parameterizer = None
        self.cond_shape = cond_shape
        super().__init__(*args, input_shape=input_shape, requires_init=True, name=name, **kwargs)

    def _build_parameterizer_fn(self, z_shape, cond_shape=0):
        """
        Builds a simple, convolutional neural network for parameterizing a distribution
        with 'num_parameters' parameters. Can be overridden by subclasses.
        """

        def Conv(rank, num_filters, kernel_size, **kwargs):
            if z_shape.rank==4:
                return Conv2D(num_filters, kernel_size, dtype=tf.float32, **kwargs)
            else:
                return Conv3D(num_filters, kernel_size, dtype=tf.float32, **kwargs)

        rank = z_shape.rank
        if z_shape.rank==4:
            x = Input((z_shape[1], z_shape[2], z_shape[3] + cond_shape), dtype=tf.float32)
        else:
            x = Input((z_shape[1], z_shape[2], z_shape[3], z_shape[4]+cond_shape), dtype=tf.float32)
        params = []
        for i in range(self.num_parameters):
            h = Conv(rank, self.num_filters, kernel_size=3, padding='same', activation='relu', 
                      kernel_initializer=RandomNormal(stddev=0.01), name=f'{self.name}/conv{rank-2}d_param{2*i}')(x)
            h = Conv(rank, self.num_filters, kernel_size=3, padding='same', activation='relu', 
                      kernel_initializer=RandomNormal(stddev=0.01), name=f'{self.name}/conv{rank-2}d_param{2*i+1}')(h)
            params.append(h)

        params = tf.concat(params, axis=-1)

        params = Conv(rank, self.num_parameters*z_shape[-1], kernel_size=3, padding='same',
                      kernel_initializer=RandomNormal(stddev=0.01), name=f'{self.name}/conv{rank-2}d')(params)
        model = Model(inputs=x, outputs=params)
        return model

    def _initialize(self, input_shape):
        if self.parameterizer is None:
            #self.log_scale = tf.Variable(tf.zeros((1,1,1,self.num_parameters*input_shape[-1])), name=f'{self.name}/log_scale')
            self.parameterizer = self._build_parameterizer_fn(input_shape, cond_shape=self.cond_shape)

    def _forward(self, x1, x2, **kwargs):
        raise NotImplementedError('missing implementation for Parameterize::_forward')

    def _inverse(self, x1, z2, **kwargs):
        raise NotImplementedError('missing implementation for Parameterize::_inverse')