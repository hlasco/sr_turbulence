import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

GAMMA = 5./3

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

def total_energy(y_true, y_pred):
    """
    Computes the Mean Square Error on the total energy
    """
    te_hr = get_total_energy(y_true)
    te_sr = get_total_energy(y_pred)

    diff = K.mean(K.square(te_hr-te_sr))
    norm = K.mean(K.square(te_hr))

    return diff/norm

def mass_flux(y_true, y_pred):
    """
    Computes the Mean Square Error on the total energy
    """
    mf_hr = get_mass_flux(y_true)
    mf_sr = get_mass_flux(y_pred)

    diff = K.mean(K.square(mf_hr-mf_sr))
    norm = K.mean(K.square(mf_hr))

    return diff/norm

def enstrophy(y_true, y_pred):
    """
    Computes the Mean Square Error on the enstrophy
    """
    ens_hr = get_enstrophy(y_true)
    ens_sr = get_enstrophy(y_pred)

    diff = K.mean(K.square(ens_hr-ens_sr))
    norm = K.mean(K.square(ens_hr))

    return diff/norm



def PSNR(y_true, y_pred):
    """
    Peek Signal to Noise Ratio
    PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
    Since input is scaled from -1 to 1, MAX_I = 1, and thus 20 * log10(1) = 0. Only the last part of the equation istherefore neccesary.
    """
    return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)

def mse(y_true, y_pred):
    #fake, real = y_pred
    #ret = -y_true * K.log(y_pred) - (1-y_true)*K.log(1-y_pred)
    return K.mean(K.square(y_pred - y_true))
    
def ResAttention(y_true, y_pred):
    target_fake = y_true[0]
    target_real = y_true[1]

    fake = y_pred[0]
    real = y_pred[1]
    #fake, real = y_pred
    ret = K.mean(K.binary_crossentropy(target_real, real) +
                 K.binary_crossentropy(target_fake, fake))
    return ret


def npGrad(inpt, dx, axis):
    return np.gradient(inpt, dx, axis=axis)
    
    
def grad_x(inpt):
    dx = 1.0/inpt.shape[1]
    ret = tf.numpy_function(npGrad, [inpt, dx,1], tf.float32)
    return ret

def grad_y(inpt):
    dx = 1.0/inpt.shape[1]
    ret = tf.numpy_function(npGrad, [inpt, dx,2], tf.float32)
    return ret

def grad_z(inpt):
    dx = 1.0/inpt.shape[1]
    ret = tf.numpy_function(npGrad, [inpt, dx,3], tf.float32)
    return ret

def get_velocity_grad(inpt):
    dudx = grad_x(inpt[:,:,:,:,0])
    dudy = grad_y(inpt[:,:,:,:,0])
    dudz = grad_z(inpt[:,:,:,:,0])

    dvdx = grad_x(inpt[:,:,:,:,1])
    dvdy = grad_y(inpt[:,:,:,:,1])
    dvdz = grad_z(inpt[:,:,:,:,1])

    dwdx = grad_x(inpt[:,:,:,:,2])
    dwdy = grad_y(inpt[:,:,:,:,2])
    dwdz = grad_z(inpt[:,:,:,:,2])

    return dudx, dvdx, dwdx, dudy, dvdy, dwdy, dudz, dvdz, dwdz

def get_vorticity(vel_grad):
    dudx, dvdx, dwdx, dudy, dvdy, dwdy, dudz, dvdz, dwdz = vel_grad
    vort_x = dwdy - dvdz
    vort_y = dudz - dwdx
    vort_z = dvdx - dudy
    return vort_x, vort_y, vort_z

def get_enstrophy(inpt):
    vel_grad = get_velocity_grad(inpt)
    vorticity = get_vorticity(vel_grad)

    omega_x, omega_y, omega_z = vorticity

    Omega = omega_x**2 + omega_y**2 + omega_z**2

    return Omega

def get_total_energy(inpt):
    rho = 10**inpt[:,:,:,:,3]
    u2  = inpt[:,:,:,:,0]**2 + inpt[:,:,:,:,1]**2 + inpt[:,:,:,:,2]**2
    ret = .5*rho*u2
    return ret

def get_mass_flux(inpt):
    rho = 10**inpt[:,:,:,:,3]
    u = inpt[:,:,:,:,0]
    v = inpt[:,:,:,:,1]
    w = inpt[:,:,:,:,2]

    ret = grad_x(rho*u) + grad_y(rho*v) + grad_z(rho*w)

    return ret
