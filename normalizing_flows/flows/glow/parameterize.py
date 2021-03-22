import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Conv3D, Lambda, Activation
from tensorflow.keras.initializers import RandomNormal
from flows.transform import Transform

class Parameterize(Transform):
    """
    Generalized base type for parameterizing a pre-specified density given some factored out latent variables.
    """
    def __init__(self, num_parameters, num_filters, input_shape=None, cond_channels=0, name='parameterize', *args, **kwargs):
        """
        Base class constructor. Should not be directly invoked by callers.

        num_parameters : number of distribution parameters per channel dimension (e.g. 2 for a Gaussian, mu and sigma)
        """
        self.num_parameters = num_parameters
        self.num_filters = num_filters
        self.parameterizer = None
        self.cond_channels = cond_channels
        super().__init__(*args, input_shape=input_shape, requires_init=True, name=name, **kwargs)

    def _build_parameterizer_fn(self, z_shape, cond_channels=0):
        """
        Builds a simple, convolutional neural network for parameterizing a distribution
        with 'num_parameters' parameters. Can be overridden by subclasses.
        """

        def Conv(dim, num_filters, kernel_size, **kwargs):
            if dim==2:
                return Conv2D(num_filters, kernel_size, dtype=tf.float32, **kwargs)
            else:
                return Conv3D(num_filters, kernel_size, dtype=tf.float32, **kwargs)

        dim = z_shape.rank -2
        shape = (*[None for _ in range(dim)], z_shape[-1] + cond_channels)
        x = Input(shape, dtype=tf.float32)
        params = []
        for i in range(self.num_parameters):
            h = Conv(dim, self.num_filters, kernel_size=3, padding='same',
                     name=f'{self.name}/conv{dim}d_param{3*i}')(x)
            h = Activation('relu')(h)
            h = Conv(dim, self.num_filters, kernel_size=3, padding='same',
                     name=f'{self.name}/conv{dim}d_param{3*i+1}')(h)
            h = Activation('relu')(h)
            h = Conv(dim, z_shape[-1], kernel_size=3, padding='same',
                      kernel_initializer='zeros', name=f'{self.name}/conv{dim}d_param{3*i+2}')(h)
            if i==0:
                h = Lambda(lambda x: tf.math.exp(self.log_scale_t)*x, dtype=tf.float32)(h)
            if i==1:
                h = Lambda(lambda x: self.steep_s*x, dtype=tf.float32)(h)
                h = Lambda(lambda x: self.scale_s*(tf.math.sigmoid(x)-.5), dtype=tf.float32)(h)
                h = Lambda(lambda x: tf.math.exp(x), dtype=tf.float32)(h)
                #h = Lambda(lambda x: 2*tf.math.sigmoid(x), dtype=tf.float32)(h)
                #h = Lambda(lambda x: tf.math.exp(x-1.0), dtype=tf.float32)(h)

            params.append(h)

        params = tf.concat(params, axis=-1)

        #params = Conv(dim, self.num_parameters*z_shape[-1], kernel_size=3, padding='same',
        #              kernel_initializer=RandomNormal(stddev=0.01), name=f'{self.name}/conv{dim}d')(params)
        model = Model(inputs=x, outputs=params)
        return model

    def _initialize(self, input_shape):
        if self.parameterizer is None:
            self.steep_s = tf.Variable(.5*tf.ones((1,1,1,input_shape[-1])), name=f'{self.name}/steep_s')
            self.scale_s = tf.Variable(.5*tf.ones((1,1,1,input_shape[-1])), name=f'{self.name}/scale_s')
            self.log_scale_t = tf.Variable(tf.zeros((1,1,1,input_shape[-1])), name=f'{self.name}/log_scale_t')
            self.parameterizer = self._build_parameterizer_fn(input_shape, cond_channels=self.cond_channels)

    def _forward(self, x1, x2, **kwargs):
        raise NotImplementedError('missing implementation for Parameterize::_forward')

    def _inverse(self, x1, z2, **kwargs):
        raise NotImplementedError('missing implementation for Parameterize::_inverse')
