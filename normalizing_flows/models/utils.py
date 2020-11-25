import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import csv
import os.path

class NormalPrior():
    def __init__(self, loc=0, scale=1.0):
        self.loc = loc
        self.scale = scale
        self.event_shape = tf.TensorShape([])

    def sample(self, shape, loc=None, scale=None):
        if loc is None:
            loc = self.loc
        if scale is None:
            scale = self.scale

        prior = tfp.distributions.Normal(loc=loc, scale=scale)
        return prior.sample(shape)

    def log_prob(self, z, loc=None, scale=None):
        if loc is None:
            loc = self.loc
        if scale is None:
            scale = self.scale

        prior = tfp.distributions.Normal(loc=loc, scale=scale)
        return prior.log_prob(z)

def update_metrics(metric_dict, **kwargs):
    for k,v in kwargs.items():
        if k in metric_dict:
            prev, n = metric_dict[k]
            metric_dict[k] = ((v + n*prev) / (n+1), n+1)
        else:
            metric_dict[k] = (v, 0)
            
def get_metrics(metric_dict):
    return {k: v[0] for k, v in metric_dict.items()}
            
def var(x: tf.Variable):
    """
    Workaround for Tensorflow bug #32748 (https://github.com/tensorflow/tensorflow/issues/32748)
    Converts tf.Variable to a Tensor via tf.identity to prevent tf.function from erroneously
    keeping weak references to garbage collected variables.
    """
    if tf.__version__ >= "2.1.0":
        return x
    else:
        return tf.identity(x)

def write_logs(metric_dict, num_step, rundir):
    metrics = get_metrics(metric_dict)
    file = rundir + '/logs.csv'

    values = list(metrics.values())
    values.insert(0, num_step)

    # Write header
    if not os.path.exists(file):
        fields = list(metrics.keys())
        fields.insert(0, 'n')
        with open(file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
    # Write values
    with open(file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(values)



def spectrum3d(fields, field_type='vel', bFac=True):
    if field_type=='vel':
        u = fields[...,0]
        v = fields[...,1]
        w = fields[...,2]
        nb, nx, ny, nz = u.shape
    elif field_type=='s':
        s = fields[:,:,:,:,-1]
        nb, nx, ny, nz = s.shape
    
    # Kill fields at boundaries to get a periodoc box
    x = tf.linspace(-np.pi/2, np.pi/2, nx)
    x, y, z = tf.meshgrid(x, x, x)
    fac = tf.math.cos(x)*tf.math.cos(y)*tf.math.cos(z)

    Kk = tf.zeros((nx,ny,nz))#( (nx//2+1, ny//2+1, nz//2+1))
    if field_type=='vel':
        for comp in [u, v, w]:
            if bFac:
                comp = comp*fac
            comp = tf.cast(comp, dtype=tf.complex64)
            tmp = (tf.signal.fftshift(tf.signal.fft3d(comp))/(nx*ny*nz))
            tmp = tf.math.abs(tf.math.square(tmp))
            Kk += 0.5*tmp
    else:
        if bFac:
            comp = fac*s
        comp = tf.cast(comp, dtype=tf.complex64)
        Kk = (tf.signal.fftshift(tf.signal.fft3d(comp))/(nx*ny*nz))
        Kk = tf.math.abs(tf.math.square(Kk))

    kx = np.arange(-nx//2,nx//2,1)

    # physical limits to the wavenumbers
    kmin = 1.0
    kmax = nx//2

    kbins = np.arange(kmin, kmax, kmin)
    N = len(kbins)

    # bin the Fourier KE into radial kbins
    kx3d, ky3d, kz3d = np.meshgrid(kx, kx, kx)#, indexing="ij")
    k = np.sqrt(kx3d**2 + ky3d**2 + kz3d**2)

    k = tf.reshape(k, [-1])
    Kk = tf.reshape(Kk, [nb, -1])
    
    #whichbin = np.digitize(k, kbins)
    whichbin = tf.raw_ops.Bucketize(input=k, boundaries=list(kbins))
    ncount = tf.math.bincount(whichbin)

    E_spectrum = []

    for n in range(len(ncount)-1):
        mask = (whichbin==n)
        masked_Kk = tf.boolean_mask(Kk, mask, axis=1)
        E_spectrum.append(tf.math.reduce_sum(masked_Kk, axis=1))

    E_spectrum=tf.stack(E_spectrum, axis=1)

    k = 0.5*(kbins[0:N-1] + kbins[1:N])
    E_spectrum = E_spectrum[:,1:N]
    return k, E_spectrum


def fft_comp(u):
    nb, nx, ny, nz = u.shape
    ret = tf.signal.fftshift(tf.signal.fft3d(u))
    ret = ret/(nx*ny*nz)
    return tf.math.abs(ret**2)
