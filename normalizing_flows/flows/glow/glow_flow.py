import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from flows.flow import Flow
from flows.transform import Transform
#from normalizing_flows.flows.affine import BatchNorm
from . import (
    InvertibleConv,
    ActNorm,
    Squeeze,
    Split,
    AffineCoupling,
    Parameterize,
    Gaussianize,
    coupling_nn_glow
)

def glow_step(layer_index, coupling_nn_ctor=coupling_nn_glow(), name='glow_step'):
    norm = ActNorm(name=f'{name}_act_norm')# if act_norm else BatchNorm(name=f'{name}_batch_norm')
    invertible_conv = InvertibleConv(name=f'{name}_inv_conv')
    affine_coupling = AffineCoupling(layer_index, nn_ctor=coupling_nn_ctor, name=f'{name}_affine_coupling')
    flow_steps = [norm, invertible_conv, affine_coupling]
    return Flow(flow_steps)

def glow_layer(layer_index,
               parameterize: Parameterize,
               depth=4,
               coupling_nn_ctor=coupling_nn_glow(),
               split_axis=-1,
               act_norm=True,
               name='glow_layer'):
    squeeze = Squeeze(name=f'{name}_squeeze')
    steps = Flow.uniform(depth, lambda i: glow_step(layer_index,
                                                    coupling_nn_ctor=coupling_nn_ctor,
                                                    name=f'{name}_step{i}'))
    layer_steps = [squeeze, steps]
    if split_axis is not None:
        layer_steps.append(Split(parameterize, split_axis=split_axis, name=f'{name}_split'))
    return Flow(layer_steps)

class GlowFlow(Transform):
    """
    Glow normalizing flow (Kingma et al, 2018).
    Note that all Glow ops define forward as x -> z (data to encoding)
    rather than the canonical interpretation of z -> x. Conversely,
    inverse is defined as z -> x (encoding to data). The implementations
    provided by this module are written to be consistent with the
    terminology as defined by the Glow authors. Note that this is inconsistent
    with the 'flows' module in general, which specifies 'forward' as z -> x
    and vice versa. This can be corrected easily using the flows.Invert transform.
    """
    def __init__(self,
                 dim=2,
                 input_channels=None,
                 num_layers=1,
                 depth=4,
                 cond_shape=None,
                 parameterize_ctor=Gaussianize,
                 coupling_nn_ctor=coupling_nn_glow(),
                 act_norm=True,
                 name='glow_flow',
                 *args, **kwargs):
        """
        Creates a new Glow normalizing flow with the given configuration.

        input_shape : shape of input; can be provided here or at a later time to 'initialize'
        num_layers  : number of "layers" in the multi-scale Glow architecture
        depth_per_layer : number of glow steps per layer

        parameterize_ctor : a function () -> Paramterize (see consructor docs for Split)
        coupling_nn_ctor : function that constructs a Keras model for affine coupling steps
        act_norm    : if true, use act norm in Glow layers; otherwise, use batch norm
        """
        def _layer(i):
            """Builds layer i; omits split op for final layer"""
            assert i < num_layers, f'expected i < {num_layers}; got {i}'
            return glow_layer(i,
                              parameterize_ctor(name=f'{name}_layer{i}_param'),
                              depth=depth,
                              coupling_nn_ctor=coupling_nn_ctor,
                              act_norm=act_norm,
                              split_axis=None if i == num_layers - 1 else -1,
                              name=f'{name}_layer{i}')
        super().__init__(*args, name=name, **kwargs)
        self.dim = dim
        self.num_layers = num_layers
        self.depth = depth
        self.cond_shape = cond_shape
        self.layers = [_layer(i) for i in range(num_layers)]
        self.parameterize = parameterize_ctor()
        if input_channels is not None:
            self.input_channels = input_channels
            input_shape = tf.TensorShape((None, *[None for _ in range(self.dim)], self.input_channels))
            self.initialize(input_shape)

    def _build_cond_fn(self, cond_shape, z_shape):
        from tensorflow.keras import Model
        from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Concatenate
        x_in = Input(z_shape[1:])
        cond_in = Input(cond_shape)
        y = Dense(tf.math.reduce_prod(z_shape[1:]), name=f'{self.name}_cond_dense')(Flatten()(cond_in))
        y = Reshape(z_shape[1:])(y)
        y = Concatenate(axis=-1)([x_in, y])
        y = Conv2D(z_shape[-1], 3, padding='same')(y)
        return Model(inputs=[x_in, cond_in], outputs=y)

    def _forward_shape(self, input_shape):
        for layer in self.layers:
            input_shape = layer._forward_shape(input_shape)
        return input_shape

    def _inverse_shape(self, input_shape):
        for layer in reversed(self.layers):
            input_shape = layer._inverse_shape(input_shape)
        return input_shape

    def _initialize(self, input_shape):
        for layer in self.layers:
            layer.initialize(input_shape)
            input_shape = layer._forward_shape(input_shape)
        self.parameterize.initialize(input_shape)
        if self.cond_shape is not None:
            self.cond_fn = self._build_cond_fn(self.cond_shape, input_shape)

    def _flatten_zs(self, zs):
        zs_reshaped = []
        for z in zs:
            zs_reshaped.append(tf.reshape(z, (tf.shape(z)[0], -1)))
        return tf.concat(zs_reshaped, axis=-1)

    def _unflatten_z(self, z):
        assert self.input_shape is not None
        shape = tf.shape(z)

        batch_size = shape[0]
        n_channels = self.input_shape[-1]

        h = int(np.ceil(np.power(shape[1]//n_channels, 1./self.dim)))
        
        ndshape = [h for _ in range(self.dim)]

        input_shape = (batch_size, *ndshape, n_channels)
        output_shape = self._forward_shape(input_shape)
        st = np.prod(output_shape[1:])
        z_k = tf.reshape(z[:,-st:], (batch_size, *output_shape[1:]))
        zs = [z_k]
        for i in range(self.num_layers-1):
            layer_i = self.layers[self.num_layers-i-1]
            output_shape = layer_i._inverse_shape(output_shape)
            size_i = np.prod(output_shape[1:])
            z_i = z[:,-st-size_i:-st]
            zs.insert(0, tf.reshape(z_i, (batch_size, *output_shape[1:])))
            st += size_i
        return zs

    def _forward(self, x, return_zs=False, **kwargs):
        assert self.cond_shape is None or 'y_cond' in kwargs, 'y_cond must be supplied for conditional flow'
        zs = []
        x_i = x
        fldj = 0.0
        for i in range(self.num_layers-1):
            layer = self.layers[i]
            (x_i, z_i), fldj_i = layer.forward(x_i)
            fldj += fldj_i
            zs.append(z_i)
        # final layer
        x_i, fldj_i = self.layers[-1].forward(x_i)
        fldj += fldj_i
        # Gaussianize (parameterize) final x_i
        h = tf.zeros_like(x_i)
        if self.cond_shape is not None:
            h = self.cond_fn([h, kwargs['y_cond']])
        z_i, fldj_i = self.parameterize.forward(h, x_i)
        fldj += fldj_i
        zs.append(z_i)
        if return_zs:
            return zs, fldj
        z = self._flatten_zs(zs)
        return tf.reshape(z, tf.shape(x)), fldj

    def _inverse(self, z, input_zs=False, **kwargs):
        assert self.cond_shape is None or 'y_cond' in kwargs, 'y_cond must be supplied for conditional flow'
        zs = z if input_zs else self._unflatten_z(tf.reshape(z, (tf.shape(z)[0], -1)))
        assert len(zs) == self.num_layers, 'number of latent space inputs should match number of layers'
        h = tf.zeros_like(zs[-1])
        if self.cond_shape is not None:
            h = self.cond_fn([h, kwargs['y_cond']])
        ildj = 0.0
        x_i, ildj_i = self.parameterize.inverse(h, zs[-1])
        ildj += ildj_i
        x_i, ildj_i = self.layers[-1].inverse(x_i)
        ildj += ildj_i
        for i in range(self.num_layers-1):
            layer = self.layers[self.num_layers-i-2]
            x_i, ildj_i = layer.inverse(x_i, zs[-i-2])
            ildj += ildj_i
        return x_i, ildj

    def _regularization_loss(self):
        return tf.math.add_n([layer._regularization_loss() for layer in self.layers])

    def param_count(self, _=None):
        ret = tf.math.reduce_sum([t.param_count(_) for t in self.layers])
        ret += self.parameterize.param_count(_)
        return ret

    def _get_trainable_variables(self, **kwargs):
        print('hi')
        if 'cond' in kwargs:
            var = self.cond_fn.trainable_variables()

        return var

    def _test(self, shape, **kwargs):
        print('Testing individual layers:')
        for i in range(self.num_layers-1):
            self.layers[i]._test(shape, **kwargs)
            shape = self.layers[i]._forward_shape(shape)

        self.layers[-1]._test(shape, **kwargs)
        shape = self.layers[-1]._forward_shape(shape)
        self.parameterize._test(shape, **kwargs)
