import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from flows.flow import Flow
from flows.transform import Transform
from flows.networks.coupling_nn import coupling_nn
from flows.networks.cond_nn import cond_nn

from tensorflow.keras.layers import UpSampling3D, AveragePooling3D
from tensorflow.keras.layers import UpSampling2D, AveragePooling2D

from . import (
    GlowFlow,
    InvertibleConv,
    ActNorm,
    Squeeze,
    Split,
    CondAffineCoupling,
    AffineInjector,
    CondSplit,
    CondGaussianize,
    Gaussianize,
    Parameterize,
)

def cglow_SR_step(depth_index, layer_index, cond_channels, upfactor, nn_ctor=coupling_nn(), name='cglow_SR_step'):

    norm = ActNorm(name=f'{name}_act_norm')

    injector = AffineInjector(layer_index, cond_channels, nn_ctor=nn_ctor,
                                name=f'{name}_affine_injector')


    if True:
        inv_conv = InvertibleConv(name=f'{name}_inv_conv')
        coupling = CondAffineCoupling(layer_index, cond_channels, nn_ctor=nn_ctor,
                                name=f'{name}_cond_affine_coupling', reverse=False)
        flow_steps = [norm, inv_conv, injector, coupling]
    else:
        coupling = CondAffineCoupling(layer_index, cond_channels, nn_ctor=nn_ctor,
                                name=f'{name}_cond_affine_coupling', reverse=True)
        flow_steps = [norm, injector, coupling]

    return Flow(flow_steps)

def cglow_SR_layer(layer_index, cond_channels, upfactor,
               parameterize: Parameterize,
               depth=4,
               nn_ctor=coupling_nn(),
               split_axis=-1,
               act_norm=True,
               name='glow_layer'):
    squeeze = Squeeze(name=f'{name}_squeeze')

    steps = Flow.uniform(depth, lambda i: cglow_SR_step(i, layer_index, cond_channels, upfactor,
                                                        nn_ctor=nn_ctor,
                                                        name=f'{name}_step{i}'))

    norm = ActNorm(name=f'{name}_trs_act_norm')
    inv_conv = InvertibleConv(name=f'{name}_trs_inv_conv')

    layer_steps = [squeeze, norm, inv_conv, steps]
    if split_axis is not None:
        layer_steps.append(CondSplit(parameterize, cond_channels=cond_channels, split_axis=split_axis, name=f'{name}_split'))
    return Flow(layer_steps)

class CGlowFlowSR(GlowFlow):
    """
    Conditional Glow normalizing flow for Super-Resolution.
    Attempts to learn a variational approximation for the joint distribution p(x,z|y)
    where x and y corresponds to HR and LR images. The conditioning input y is first processed by the
    encoding network defined in cond_fn, then used in the flow operations cond_coupling, injector,
    and cond_gaussianize.
    """
    def __init__(self,
                 dim=2,
                 input_channels=None,
                 upfactor=1,
                 num_layers=1,
                 depth=4,
                 parameterize_ctor=Gaussianize(),
                 coupling_ctor=coupling_nn(),
                 cond_ctor=cond_nn(),
                 act_norm=True,
                 name='cglow_flow',
                 *args, **kwargs):
        """
        Creates a new Glow normalizing flow with the given configuration.
        dim            : the spatial dimension of the input x
        input_channels : the channel dimension of the input x;
                         can be provided here or at a later time to 'initialize'
        num_layers     : number of "layers" in the multi-scale Glow architecture
        depth          : number of glow steps per layer
        parameterize_ctor       : a function () -> Paramterize (see consructor docs for Split)
        coupling_ctor : function that constructs a Keras model for the coupling networks
        cond_ctor     : function that constructs a Keras model for conditioning network
        act_norm    : if true, use act norm in Glow layers; otherwise, use batch norm
        """
        self.dim = dim

        self.cond_fn = cond_ctor()
        self.cond_channels = self.cond_fn.outputs[0].shape[-1]

        def _layer(i):
            """Builds layer i; omits split op for final layer"""
            assert i < num_layers, f'expected i < {num_layers}; got {i}'
            return cglow_SR_layer(i,self.cond_channels, upfactor,
                              parameterize_ctor(i=i, name=f'{name}_layer{i}_param',
                                                cond_channels=self.cond_channels),
                              depth=depth,
                              nn_ctor=coupling_ctor,
                              act_norm=act_norm,
                              split_axis=None if i == num_layers - 1 else -1,
                              name=f'{name}_layer{i}')

        Transform.__init__(self, *args, name=name, **kwargs)
        self.num_layers = num_layers
        self.depth = depth

        self.upfactor = upfactor
        self.parameterize = parameterize_ctor(i=num_layers-1, cond_channels=self.cond_channels)

        self.layers = [_layer(i) for i in range(num_layers)]

        if input_channels is not None:
            self.input_channels = input_channels
            input_shape = tf.TensorShape((None, *[None for i in range(self.dim)], self.input_channels))
            self.initialize(input_shape)

    def _initialize(self, input_shape):
        for layer in self.layers:
            layer.initialize(input_shape)
            input_shape = layer._forward_shape(input_shape)
        self.parameterize.initialize(input_shape)

    def _forward(self, x, return_zs=False, **kwargs):
        assert 'y_cond' in kwargs, 'y_cond must be supplied for conditional flow'
        zs = []
        x_i = x
        fldj = 0.0

        u = self.cond_fn(kwargs['y_cond'])

        for i in range(self.num_layers-1):
            layer = self.layers[i]
            (x_i, z_i), fldj_i = layer.forward(x_i, y_cond=u[i])
            fldj += fldj_i
            zs.append(z_i)
        # final layer

        x_i, fldj_i = self.layers[-1].forward(x_i, y_cond=u[-1])
        fldj += fldj_i
        # Gaussianize (parameterize) final x_i
        h = tf.zeros_like(x_i)
        z_i, fldj_i = self.parameterize.forward(h, x_i, y_cond=u[-1])
        fldj += fldj_i
        zs.append(z_i)
        if return_zs:
            return zs, fldj
        z = self._flatten_zs(zs)
        return tf.reshape(z, tf.shape(x)), fldj

    def _inverse(self, z, input_zs=False, **kwargs):
        assert 'y_cond' in kwargs, 'y_cond must be supplied for conditional flow'
        zs = z if input_zs else self._unflatten_z(tf.reshape(z, (tf.shape(z)[0], -1)))
        assert len(zs) == self.num_layers, 'number of latent space inputs should match number of layers'
        h = tf.zeros_like(zs[-1])

        u = self.cond_fn(kwargs['y_cond'])

        ildj = 0.0
        x_i, ildj_i = self.parameterize.inverse(h, zs[-1], y_cond=u[-1])
        ildj += ildj_i

        x_i, ildj_i = self.layers[-1].inverse(x_i, y_cond=u[-1])
        ildj += ildj_i
        for i in range(self.num_layers-1):
            i_inv = self.num_layers-i-2
            layer = self.layers[i_inv]
            x_i, ildj_i = layer.inverse(x_i, zs[i_inv], y_cond=u[i_inv])
            ildj += ildj_i
        return x_i, ildj

    def _regularization_loss(self):
        return tf.math.add_n([layer._regularization_loss() for layer in self.layers])

    def param_count(self, _=None):
        ret = tf.math.reduce_sum([t.param_count(_) for t in self.layers])
        ret += self.parameterize.param_count(_)
        ret += self.cond_fn.count_params()
        return ret

    def _test(self, shape, **kwargs):
        print('Testing individual layers:')
        for i in range(self.num_layers-1):
            self.layers[i]._test(shape, **kwargs)
            shape = self.layers[i]._forward_shape(shape)

        self.layers[-1]._test(shape, **kwargs)
        shape = self.layers[-1]._forward_shape(shape)
        self.parameterize._test(shape, **kwargs)
        print('\t Num params:', self.parameterize.param_count(shape))

    def _get_trainable_variables(self, **kwargs):
        ret = {}
        ret['cond'] = self.cond_fn.trainable_variables

        var = []
        for l in self.layers:
            var += l.trainable_variables
        var += self.parameterize.trainable_variables
        ret['flow'] = var
        return ret
