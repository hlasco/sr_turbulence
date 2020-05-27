import tensorflow as tf
from tensorflow.keras import backend as K

def pixel(y_true, y_pred):
    """
    Computes the Mean Square Error on the pixels of each channels
    """
    diff = K.mean(K.square(y_true-y_pred))
    norm = K.mean(K.square(y_true))
    loss = diff / norm
    return loss

def grad(y_true,y_pred):
    """
    Computes the Mean Square Error on the gradients of each channels
    """
    grad_hr_x = grad_x(y_true)
    grad_hr_y = grad_y(y_true)
    grad_hr_z = grad_z(y_true)
    grad_sr_x = grad_x(y_pred)
    grad_sr_y = grad_y(y_pred)
    grad_sr_z = grad_z(y_pred)

    grad_diff = 1./3*K.mean(K.square(grad_hr_x-grad_sr_x)) + \
                1./3*K.mean(K.square(grad_hr_y-grad_sr_y)) + \
                1./3*K.mean(K.square(grad_hr_z-grad_sr_z))
    grad_norm = 1./3*K.mean(K.square(grad_hr_x)) + \
                1./3*K.mean(K.square(grad_hr_y)) + \
                1./3*K.mean(K.square(grad_hr_z))

    grad_loss = grad_diff / grad_norm

    return grad_loss

def PSNR(y_true, y_pred):
    """
    Peek Signal to Noise Ratio
    PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
    Since input is scaled from -1 to 1, MAX_I = 1, and thus 20 * log10(1) = 0. Only the last part of the equation istherefore neccesary.
    """
    return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)
    
    
    
def ResAttention(y_true, y_pred):
    fake_logit = y_pred[0]
    real_logit = y_pred[1]
    ret = K.mean(K.binary_crossentropy(K.zeros_like(fake_logit), fake_logit) +
                 K.binary_crossentropy(K.ones_like(real_logit), real_logit))
    return ret



    
    
def grad_x(input):
    out = []
    nChannels = input.shape[-1]
    for i in range(nChannels):
        out.append(ddx(input,i))
    ret = tf.stack(out, axis=-1)
    return ret

def grad_y(input):
    out = []
    nChannels = input.shape[-1]
    for i in range(nChannels):
        out.append(ddy(input,i))
    ret = tf.stack(out, axis=-1)
    return ret

def grad_z(input):
    out = []
    nChannels = input.shape[-1]
    for i in range(nChannels):
        out.append(ddz(input,i))
    ret = tf.stack(out, axis=-1)
    return ret

def ddx(inpt, channel):
    inpt_shape = inpt.get_shape().as_list()
    var = tf.expand_dims( inpt[:,:,:,:,channel], axis=4 )

    ddx1D = tf.constant([-1./60., 3./20., -3./4., 0., 3./4., -3./20., 1./60.], dtype=tf.float32)
    ddx3D = tf.reshape(ddx1D, shape=(-1,1,1,1,1))

    strides = [1,1,1,1,1]
    output = tf.nn.conv3d(var, ddx3D, strides, padding = 'VALID', data_format = 'NDHWC', name=None)
    return output

def ddy(inpt, channel):
    inpt_shape = inpt.get_shape().as_list()
    var = tf.expand_dims( inpt[:,:,:,:,channel], axis=4 )

    ddx1D = tf.constant([-1./60., 3./20., -3./4., 0., 3./4., -3./20., 1./60.], dtype=tf.float32)
    ddx3D = tf.reshape(ddx1D, shape=(1,-1,1,1,1))

    strides = [1,1,1,1,1]
    output = tf.nn.conv3d(var, ddx3D, strides, padding = 'VALID', data_format = 'NDHWC', name=None)
    return output

def ddz(inpt, channel):
    inpt_shape = inpt.get_shape().as_list()
    var = tf.expand_dims( inpt[:,:,:,:,channel], axis=4 )

    ddx1D = tf.constant([-1./60., 3./20., -3./4., 0., 3./4., -3./20., 1./60.], dtype=tf.float32)
    ddx3D = tf.reshape(ddx1D, shape=(1,1,-1,1,1))

    strides = [1,1,1,1,1]
    output = tf.nn.conv3d(var, ddx3D, strides, padding = 'VALID', data_format = 'NDHWC', name=None)
    return output
