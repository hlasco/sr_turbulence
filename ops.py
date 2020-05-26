import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
import h5py as h5
import matplotlib.pyplot as plt
import os, io, glob, sys

#import horovod.tensorflow.keras as hvd
import numpy as np


def MSE(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def MAE(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def grad_x(input, nChannels):
    out = []
    for i in range(nChannels):
        out.append(ddx(input,i))
    ret = tf.stack(out, axis=-1)
    return ret

def grad_y(input, nChannels):
    out = []
    for i in range(nChannels):
        out.append(ddy(input,i))
    ret = tf.stack(out, axis=-1)
    return ret

def grad_z(input, nChannels):
    out = []
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

class DataLoader():
    """
    Tool for loading snapshots frpm a directory
    """

    def __init__(self,
                 LR_directory,      # Directory containing the High-res snapshots
                 HR_directory,      # Directory containing the High-res snapshots
                 batch_size=4,     # Batch size used for training
                 HR_boxsize=128,    # High-res full box size
                 sr_factor=4,       # Super-resolution factor
                 HR_patchsize=64,   # High-res patch size
                 nChannels=5,       # Number of channels (u,v,w,rho,P)
                 ):

        self.HR_directory = HR_directory
        self.LR_directory = LR_directory
        self.batch_size = batch_size
        self.HR_boxsize = HR_boxsize
        self.sr_factor = sr_factor
        self.HR_patchsize = HR_patchsize
        self.nChannels = nChannels

        self.channels = ['velocity_x','velocity_y','velocity_z','density','pressure']
        #if hvd.rank() == 0:
        print('High-res data ', HR_directory+'/*.h5')
        print('Low-res data ', LR_directory+'/*.h5')

        self.HR_fList = sorted(glob.glob(HR_directory+'/*.h5'))
        self.HR_nFile = len(self.HR_fList)

        self.LR_fList = sorted(glob.glob(LR_directory+'/*.h5'))
        self.nFiles = len(self.LR_fList)

        if self.nFiles is not len(self.HR_fList):
            raise ValueError('HR and LR directories should contain the same number of snapshots \
                              LR_nFile={}, HR_nFile={}'.format(self.nFiles, len(self.HR_fList)))


    def loadRandomBatch(self):
        #if hvd.rank()==0:
        print('>> Loading a random batch of size {}'.format(self.batch_size), flush=True)
        ps_lr = self.HR_patchsize // self.sr_factor
        ps_hr = self.HR_patchsize
        LR_train = np.zeros([self.batch_size,ps_lr,ps_lr,ps_lr,self.nChannels] )
        HR_train = np.zeros([self.batch_size,ps_hr,ps_hr,ps_hr,self.nChannels] )
        nPatches = self.HR_boxsize // self.HR_patchsize
        snap_ids = np.random.randint(0, self.nFiles, self.batch_size) # Here I can split train/test :)

        # For each random snapshot, pick a random cube of size (LR/HR)_boxsize**3
        for bNum, snap_id in enumerate(snap_ids):
            bIDX, bIDY, bIDZ = np.random.randint(0, nPatches, 3)
            LR_file = self.LR_fList[snap_id]
            HR_file = self.HR_fList[snap_id]

            with h5.File(LR_file, 'r') as f:
                gas = f['gas']
                info = f['info']
                dims = info.attrs['dims']
                for cNum, channel in enumerate(self.channels):
                    patch = np.array(gas[channel]).reshape(dims)
                    patch = patch[bIDX*ps_lr:(bIDX+1)*ps_lr,
                                  bIDY*ps_lr:(bIDY+1)*ps_lr,
                                  bIDZ*ps_lr:(bIDZ+1)*ps_lr]
                    if channel in ['density', 'pressure']:
                        patch = np.log(patch)
                    mu = np.mean(patch)
                    std = np.std(patch)
                    LR_train[bNum,:,:,:,cNum]=(patch-mu)/std

            with h5.File(HR_file, 'r') as f:
                gas = f['gas']
                info = f['info']
                dims = info.attrs['dims']
                for cNum, channel in enumerate(self.channels):
                    patch = np.array(gas[channel]).reshape(dims)
                    patch = patch[bIDX*ps_hr:(bIDX+1)*ps_hr,
                                  bIDY*ps_hr:(bIDY+1)*ps_hr,
                                  bIDZ*ps_hr:(bIDZ+1)*ps_hr]
                    if channel in ['density', 'pressure']:
                        patch = np.log(patch)
                    mu = np.mean(patch)
                    std = np.std(patch)
                    HR_train[bNum,:,:,:,cNum]=(patch-mu)/std
        #if hvd.rank()==0:
        print('>>   Done', flush=True)
        return LR_train, HR_train


def Image_generator(box1,box2,box3):
    fig,axs = plt.subplots(1,3)
    axs[0].imshow(box1)
    axs[0].set_title('Filtered')
    axs[1].imshow(box2)
    axs[1].set_title('PISRT_GAN')
    axs[2].imshow(box3)
    axs[2].set_title('Unfiltered')
    plt.show()

class ImgCallback(Callback):
    def __init__(self,logpath, generator, lr_data, hr_data):
        self.generator = generator
        self.lr_data = lr_data
        self.hr_data = hr_data
        self.logpath=logpath

    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def image_grid(self, lr, sr, hr):
        # Create a figure to contain the plot.

        nC = lr.shape[-1]
        figure, ax = plt.subplots(3, nC, figsize=(8,5))
        titles=[r'$u$',r'$v$',r'$w$',r'$\log \rho$',r'$\log P$']
        cmaps = [plt.cm.RdBu, plt.cm.RdBu, plt.cm.RdBu, plt.cm.viridis,plt.cm.viridis]
        ax[0,0].set_ylabel('Filtered')
        ax[1,0].set_ylabel('PISRT_GAN')
        ax[2,0].set_ylabel('Unfiltered')
        for i in range(nC):
            vmin = np.min(hr[:,:,i])
            vmax = np.max(hr[:,:,i])
            for j in range(3):
                ax[j,i].set_xticks([])
                ax[j,i].set_yticks([])
                ax[j,i].grid(False)
                if j==0:
                    ax[j,i].set_title(titles[i])
                    ax[j,i].imshow(lr[:,:,i], cmap=cmaps[i], vmin=vmin, vmax=vmax)
                if j==1:
                    ax[j,i].imshow(sr[:,:,i], cmap=cmaps[i], vmin=vmin, vmax=vmax)
                if j==2:
                    ax[j,i].imshow(hr[:,:,i], cmap=cmaps[i], vmin=vmin, vmax=vmax)
        return figure

    def on_batch_end(self, epoch, logs={}):
        lr = self.lr_data
        hr = self.hr_data

        sr = self.generator.predict(lr, steps=1)

        lr = lr[0,:,:,0,:]
        sr = sr[0,:,:,0,:]
        hr = hr[0,:,:,0,:]
        figure = self.image_grid(lr, sr, hr)
        file_writer = tf.summary.create_file_writer(self.logpath)
        with file_writer.as_default():
            tf.summary.image('Channel Slices', self.plot_to_image(figure), step=0)
        return


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()
