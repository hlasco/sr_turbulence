import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Conv3D, Concatenate, Activation, add, Lambda, Add, LeakyReLU
from tensorflow.keras.layers import UpSampling2D, UpSampling3D


def cond_nn(upfactor=2, num_layers=3, dim=2, cond_channels=3, cond_filters=64, kernel_size=3, cond_blocks=4, cond_resblocks=12):
    w_init = tf.random_normal_initializer(stddev=0.02)
    
    resize_factors = [upfactor - (i+1) for i in range(num_layers)]

    def Conv(dim, num_filters, kernel_size, **kwargs):
        if dim==2:
            return Conv2D(num_filters, kernel_size, padding='same', dtype=tf.float32, **kwargs)
        else:
            return Conv3D(num_filters, kernel_size, padding='same', dtype=tf.float32, **kwargs)
        
    def Upsample(dim, size=2):
        if dim==2:
            return UpSampling2D(size=size)
        else:
            return UpSampling3D(size=size)
        
    def resize_cond(x, dim, fac, num_filters, kernel_size, base_name):
        if fac==0:
            x = Conv(dim, num_filters, kernel_size, name=f'{base_name}/conv{dim}d_0')(x)
            return x
        if fac > 0:
            for i in range(fac-1):
                x = Upsample(dim, size=2)(x)
                x = Conv(dim, num_filters, kernel_size, name=f'{base_name}/conv{dim}d_{i}')(x)
                x = LeakyReLU(0.2)(x)
            x = Upsample(dim, size=2)(x)
            x = Conv(dim, num_filters, kernel_size, name=f'{base_name}/conv{dim}d_{fac-1}')(x)
            return x
        if fac < 0:
            for i in range(-fac-1):
                x = Conv(dim, num_filters, kernel_size, strides=2, name=f'{base_name}/conv{dim}d_{i}')(x)
                x = LeakyReLU(0.2)(x)
            x = Conv(dim, num_filters, kernel_size, strides=2, name=f'{base_name}/conv{dim}d_{-fac-1}')(x)
            return x

    def _resnet_block(x, dim, num_filters, num_resblocks, base_name):
        u = x
        h = x
        for i in range(cond_resblocks):
            u = Conv(dim, num_filters, kernel_size, kernel_initializer=w_init, name=f'{base_name}/conv{dim}d_{i}')(h)
            
            if i<num_resblocks-1:
                u = LeakyReLU(0.2)(u)
                h = Concatenate(axis=-1)([h,u])
        output = add([u, .2*x])
        return output

    def f():
        shape = (*[None for _ in range(dim)], cond_channels)
        y = Input(shape, dtype=tf.float32)

        u_pre = Conv(dim, cond_filters, kernel_size, kernel_initializer=w_init)(y)

        u = _resnet_block(u_pre, dim, cond_filters, cond_resblocks, f'cond_block_0')
        outputs_raw = u
        for i in range(1, cond_blocks):
            u = _resnet_block(u, dim, cond_filters, cond_resblocks, f'cond_block_{i}')
            outputs_raw = Concatenate(axis=-1)([outputs_raw, u])
        outputs_raw = Concatenate(axis=-1)([outputs_raw, u_pre])
        
        n_filt_raw = cond_filters*(1+cond_blocks)
        
        outputs=[resize_cond(outputs_raw, dim, f, n_filt_raw, kernel_size, f'resize_cond_{f}') for f in resize_factors]
 
        model = Model(inputs=y, outputs=outputs, name='cond_fn')
        return model
    return f
