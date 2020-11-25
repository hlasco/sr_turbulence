import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Conv3D, Concatenate, Activation, add, Lambda, Add

def cond_nn(dim=2, cond_channels=3, cond_filters=64, kernel_size=3, cond_blocks=4, cond_resblocks=12):
    w_init = tf.random_normal_initializer(stddev=0.02)
    def Conv(dim, num_filters, kernel_size, **kwargs):
        if dim==2:
            return Conv2D(num_filters, kernel_size, dtype=tf.float32, **kwargs)
        else:
            return Conv3D(num_filters, kernel_size, dtype=tf.float32, **kwargs)

    def _resnet_block(x, dim, num_filters, num_resblocks, base_name):
        u = x
        h = x
        for i in range(cond_resblocks):
            u = Conv(dim, cond_filters, kernel_size, padding='same', kernel_initializer=w_init, name=f'{base_name}/conv{dim}d_{i}')(h)
            u = Activation('relu')(u)
            if i<num_resblocks-1:
                h = Concatenate(axis=-1)([h,u])
        output = add([u, .2*x])
        return output
    def f():
        shape = (*[None for _ in range(dim)], cond_channels)
        y = Input(shape, dtype=tf.float32)
        u_pre = Conv(dim, cond_filters, kernel_size, padding='same', kernel_initializer=w_init)(y)
        u = _resnet_block(u_pre, dim, cond_filters, cond_resblocks, f'cond_block_0')
        output = u
        for i in range(1, cond_blocks):
            u = _resnet_block(u, dim, cond_filters, cond_resblocks, f'cond_block_{i}')
            output = Concatenate(axis=-1)([output, u])
        output = Concatenate(axis=-1)([output, u_pre])
        model = Model(inputs=y, outputs=output, name='cond_fn')
        return model
    return f
