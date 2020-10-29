import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from flows.flow import Flow
from flows.transform import Transform
#from normalizing_flows.flows.affine import BatchNorm
from . import (
    GlowFlow,
    InvertibleConv,
    ActNorm,
    Squeeze,
    Split,
    CondAffineCoupling,
    AffineCoupling,
    AffineInjector,
    CondSplit,
    Gaussianize,
    cond_gaussianize,
    Parameterize,
    cond_coupling_nn_glow,
    coupling_nn_glow,
    injector_nn_glow,
)

def getMesh(dim, s, x_min, x_max):
    x0, x1, x2 = tf.meshgrid(
        tf.linspace(x_min[0], x_max[0], num=s[0]),
        tf.linspace(x_min[1], x_max[1], num=s[1]),
        tf.linspace(x_min[2], x_max[2], num=s[2]),
        indexing='ij')
    return x0, x1, x2

def reshape_volume(inpt, output_shape):
    s = inpt.shape
    new_s = [s[0], output_shape[0], output_shape[1], output_shape[2], s[4]]
    dim = len(output_shape)

    x_min = [0. for i in range(dim)]
    x_max = [1. for i in range(dim)]

    x0, x1, x2 = getMesh(dim, s[1:-1], x_min, x_max)
    y0, y1, y2 = getMesh(dim, output_shape, x_min, x_max)

    y = tf.stack([y0,y1,y2], axis=-1)
    y = tf.reshape(y, (-1,3))

    ret = tfp.math.batch_interp_regular_nd_grid(y, x_min, x_max, inpt, axis=-4)
    ret = tf.reshape(ret, shape=new_s)
    return ret

@tf.function
def reshape_cond(y_cond, i_layer, upfactor):
    shape = tf.shape(y_cond)
    newshape = shape[1:-1]
    fac =  2.**upfactor/2**(i_layer+1)
    dim = y_cond.shape.rank-2
    if fac==1:
        return y_cond
    elif fac>1:
        fac = int(fac)
        newshape = tf.cast(fac * shape[1:-1], dtype=tf.int32)
    elif fac<1:
        fac = int(1./fac)
        newshape = tf.cast(shape[1:-1]//fac, dtype=tf.int32)
    if dim==2:
        y_reshaped = tf.image.resize(y_cond, newshape)
    else:
        y_reshaped = reshape_volume(y_cond, newshape)
    return y_reshaped

def cglow_SR_step(depth_index, layer_index, cond_shape, upfactor,
                  affine_coupling_nn_ctor=coupling_nn_glow(),
                  cond_coupling_nn_ctor=cond_coupling_nn_glow(),
                  injector_nn_ctor=injector_nn_glow(),
                  name='cglow_SR_step'):

    norm = ActNorm(name=f'{name}_act_norm')

    injector = AffineInjector(layer_index, cond_shape, nn_ctor=injector_nn_ctor,
                                name=f'{name}_affine_injector')


    if layer_index < 2 :
        inv_conv = InvertibleConv(name=f'{name}_inv_conv')
        coupling = CondAffineCoupling(layer_index, cond_shape, nn_ctor=cond_coupling_nn_ctor,
                                name=f'{name}_cond_affine_coupling', reverse=False)
        flow_steps = [norm, inv_conv, coupling, injector]
    else:
        coupling = CondAffineCoupling(layer_index, cond_shape, nn_ctor=cond_coupling_nn_ctor,
                                name=f'{name}_cond_affine_coupling', reverse=True)
        flow_steps = [norm, coupling, injector]

    return Flow(flow_steps)

def cglow_SR_layer(layer_index, cond_shape, upfactor,
               parameterize: Parameterize,
               depth=4,
               affine_coupling_nn_ctor=coupling_nn_glow(),
               cond_coupling_nn_ctor=cond_coupling_nn_glow(),
               injector_nn_ctor=injector_nn_glow(),
               split_axis=-1,
               act_norm=True,
               name='glow_layer'):
    squeeze = Squeeze(name=f'{name}_squeeze')

    steps = Flow.uniform(depth, lambda i: cglow_SR_step(i, layer_index, cond_shape, upfactor,
                                                    affine_coupling_nn_ctor=affine_coupling_nn_ctor,
                                                    cond_coupling_nn_ctor=cond_coupling_nn_ctor,
                                                    injector_nn_ctor=injector_nn_ctor,
                                                    name=f'{name}_step{i}'))

    norm = ActNorm(name=f'{name}_act_norm')
    inv_conv = InvertibleConv(name=f'{name}_inv_conv')

    layer_steps = [squeeze, norm, inv_conv, steps]
    if split_axis is not None:
        layer_steps.append(CondSplit(parameterize, cond_shape=cond_shape, split_axis=split_axis, name=f'{name}_split'))
    return Flow(layer_steps)

class CGlowFlowSR(GlowFlow):
    """
    Conditional Glow normalizing flow for Super-Resolution.
    Attempts to learn a variational approximation for the joint distribution p(x,z|y)
    where x and y corresponds to HR and LR images. The conditioning input y is first processed by the
    encoding network defined in _build_cond_fn, then used in the flow operations cond_coupling, injector,
    and cond_gaussianize.
    """
    def __init__(self,
                 dim=2,
                 input_channels=None,
                 upfactor=1,
                 num_layers=1,
                 depth=4,
                 cond_channels=None,
                 cond_filters=64,
                 cond_resblocks=12,
                 cond_blocks=4,
                 parameterize_ctor =cond_gaussianize(),
                 affine_coupling_nn_ctor=coupling_nn_glow(),
                 cond_coupling_nn_ctor=cond_coupling_nn_glow(),
                 injector_nn_ctor=injector_nn_glow(),
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
        cond_channels  : the channel dimension of the conditional input y. At initialization,
                         this dimension is updated to the channel dimension of ENCODED contitional input
                         cond_fn(y)
        cond_filters   : number of filters in the encoding network
        cond_resbkocks : number of residual blocks in the encoding network
        cond_blocks    : number of bocks in the encoding network
        parameterize_ctor       : a function () -> Paramterize (see consructor docs for Split)
        affine_coupling_nn_ctor : function that constructs a Keras model for affine coupling steps
        cond_coupling_nn_ctor   : function that constructs a Keras model for conditional coupling steps
        injector_nn_ctor        : function that constructs a Keras model for injector steps
        act_norm    : if true, use act norm in Glow layers; otherwise, use batch norm
        """
        self.dim = dim
        self.cond_channels = cond_channels
        if self.cond_channels is not None:
            cond_shape = tf.TensorShape((None, *[None for i in range(self.dim)], self.cond_channels))
            self.cond_fn = self._build_cond_fn(cond_shape, num_filters=cond_filters,
                                 num_blocks=cond_blocks, num_resblocks=cond_resblocks)
            self.cond_shape = self.cond_fn.layers[-1].output_shape[-1]

        def _layer(i):
            """Builds layer i; omits split op for final layer"""
            assert i < num_layers, f'expected i < {num_layers}; got {i}'
            return cglow_SR_layer(i,self.cond_shape, upfactor,
                              parameterize_ctor(i=i, name=f'{name}_layer{i}_param',
                                                cond_shape=self.cond_shape),
                              depth=depth,
                              affine_coupling_nn_ctor=affine_coupling_nn_ctor,
                              cond_coupling_nn_ctor=cond_coupling_nn_ctor,
                              injector_nn_ctor=injector_nn_ctor,
                              act_norm=act_norm,
                              split_axis=None if i == num_layers - 1 else -1,
                              name=f'{name}_layer{i}')

        Transform.__init__(self, *args, name=name, **kwargs)
        self.num_layers = num_layers
        self.depth = depth

        self.upfactor = upfactor
        self.parameterize = parameterize_ctor(i=num_layers-1, cond_shape=self.cond_shape)

        self.layers = [_layer(i) for i in range(num_layers)]

        if input_channels is not None:
            self.input_channels = input_channels
            input_shape = tf.TensorShape((None, *[None for i in range(self.dim)], self.input_channels))
            self.initialize(input_shape)

    def _build_cond_fn(self, cond_shape, num_filters=64, kernel_size=3, num_blocks=4, num_resblocks=12):
        from tensorflow.keras import Model
        from tensorflow.keras.layers import Input, Conv2D, Conv3D, Concatenate, Activation, add, Lambda, Add
        def Conv(dim, num_filters, kernel_size, **kwargs):
            if dim==2:
                return Conv2D(num_filters, kernel_size, dtype=tf.float32, **kwargs)
            else:
                return Conv3D(num_filters, kernel_size, dtype=tf.float32, **kwargs)

        def _resnet_block(x, dim, num_filters, num_resblocks, base_name):
            h = x
            for i in range(num_resblocks):
                h = Activation('relu')(h)
                h = Conv(dim, num_filters, kernel_size, padding='same', name=f'{base_name}/conv{dim}d_{i}')(h)
            h = add([x, h])
            return h

        dim = cond_shape.rank-2
        y = Input(cond_shape[1:])

        u_pre = Conv(dim, num_filters, kernel_size, padding='same')(y)

        u = _resnet_block(u_pre, dim, num_filters, num_resblocks, f'cond_block_0')
        #output = u
        for i in range(1, num_blocks):
            u = _resnet_block(u, dim, num_filters, num_resblocks, f'cond_block_{i}')
            #output = Concatenate(axis=-1)([output, u])

        #output = Concatenate(axis=-1)([output, u_pre])
        model = Model(inputs=y, outputs=u, name='cond_fn')
        return model

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
            u_reshaped = reshape_cond(u, i, self.upfactor)
            (x_i, z_i), fldj_i = layer.forward(x_i, y_cond=u_reshaped)
            fldj += fldj_i
            zs.append(z_i)
        # final layer

        u_reshaped = reshape_cond(u, self.num_layers-1, self.upfactor)
        x_i, fldj_i = self.layers[-1].forward(x_i, y_cond=u_reshaped)
        fldj += fldj_i
        # Gaussianize (parameterize) final x_i
        h = tf.zeros_like(x_i)
        z_i, fldj_i = self.parameterize.forward(h, x_i, y_cond=u_reshaped)
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
        u_reshaped = reshape_cond(u, self.num_layers-1, self.upfactor)
        x_i, ildj_i = self.parameterize.inverse(h, zs[-1], y_cond=u_reshaped)
        ildj += ildj_i

        x_i, ildj_i = self.layers[-1].inverse(x_i, y_cond=u_reshaped)
        ildj += ildj_i
        for i in range(self.num_layers-1):
            layer = self.layers[self.num_layers-i-2]
            u_reshaped = reshape_cond(u, self.num_layers-i-2, self.upfactor)
            x_i, ildj_i = layer.inverse(x_i, zs[-i-2], y_cond=u_reshaped)
            ildj += ildj_i
        return x_i, ildj

    def _regularization_loss(self):
        return tf.math.add_n([layer._regularization_loss() for layer in self.layers])

    def param_count(self, _=None):
        ret = tf.math.reduce_sum([t.param_count(_) for t in self.layers])
        ret += self.parameterize.param_count(_)
        ret += self.cond_fn.count_params()
        print(self.cond_fn.count_params())
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
