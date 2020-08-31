import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Lambda, Layer
import h5py as h5
import matplotlib.pyplot as plt
import os, io, glob, sys

#import horovod.tensorflow.keras as hvd
import numpy as np

class ReflectionPadding3D(Layer):
    def __init__(self, padding=(1, 1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=5)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def get_output_shape_for(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0],
                      s[1] + 2 * self.padding[0],
                      s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad,d_pad = self.padding
        return tf.pad(x, [[0,0,0],
                          [h_pad,h_pad,h_pad],
                          [w_pad,w_pad,w_pad],
                          [d_pad,d_pad,d_pad], [0,0,0] ], 'REFLECT')


def PixelShuffling(input_shape, name, scale=2):
    """
    Layer to do 3D pixel shuffling.
    :param input_shape: tensor shape, (batch, height, width, depth, channel)
    :param scale: upsampling scale. Default=2
    :return:
    """

    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                input_shape[1] * scale,
                input_shape[2] * scale,
                input_shape[3] * scale,
                int(input_shape[4] / (scale ** 3))]
        output_shape = tuple(dims)
        return output_shape
    
    def phaseShift(inputs, shape_1, shape_2):
        # Tackle the condition when the batch is None
        X = tf.reshape(inputs, shape_1)
        X = tf.transpose(X, [0, 1, 4, 2, 5, 3, 6])

        return tf.reshape(X, shape_2)

    def subpixel(x):
        size = x.get_shape().as_list()
        size1 = tf.shape(x)
        batch_size = -1
        d = size[1]
        h = size[2]
        w = size[3]
        c = size[4]

        # Get the target channel size
        channel_target = c // (scale * scale * scale)
        channel_factor = c // channel_target

        shape_1 = [batch_size, d, h, w, scale, scale, scale]
        shape_2 = [batch_size, d * scale, h * scale, w * scale, 1]
        # Reshape and transpose for periodic shuffling for each channel
        input_split = tf.split(x, channel_target, axis=4)
        ret = tf.concat([phaseShift(split, shape_1, shape_2) for split in input_split], axis=4)
        return ret

    return Lambda(subpixel, output_shape=subpixel_shape, name=name)

class DataLoader():
    """
    Tool for loading snapshots from a directory
    """

    def __init__(self,
                 directory,         # Directory containing training and validation data
                 hr_boxsize=256,    # High-res full box size
                 sr_factor=4,       # Super-resolution factor
                 hr_patchsize=64,   # High-res patch size
                 nChannels=4,       # Number of channels (u,v,w,log10rho)
                 ):

        self.directory    = directory
        self.hr_boxsize   = hr_boxsize
        self.sr_factor    = sr_factor
        self.hr_patchsize = hr_patchsize
        self.nChannels    = nChannels

        # Just to read my hdf5 snapshots
        self.channels = ['ux','uy','uz','rho']

        # Make sure that your file names can be sorted
        self.fList_t  = glob.glob(directory+'/train/*/processed_data/snapshot.h5')
        self.nFiles_t = len(self.fList_t)

        self.fList_v  = glob.glob(directory+'/validation/*/processed_data/snapshot.h5')
        self.nFiles_v = len(self.fList_v)

        print(self.nFiles_t, self.nFiles_v)

    def loadRandomBatch(self, batch_size, bTrain=True):
        """
        Load a batch of data for training.
        """
        if bTrain:
            fList  = self.fList_t
            nFiles = self.nFiles_t
        else:
            fList  = self.fList_v
            nFiles = self.nFiles_v


        ps_lr = self.hr_patchsize // self.sr_factor
        ps_hr = self.hr_patchsize

        lr_train = np.zeros([batch_size,ps_lr,ps_lr,ps_lr,self.nChannels] )
        hr_train = np.zeros([batch_size,ps_hr,ps_hr,ps_hr,self.nChannels] )

        # Number of snapshots needed to fill the batch
        nPatches = self.hr_boxsize // self.hr_patchsize

        snap_ids = np.random.randint(0, nFiles, batch_size)

        # For each random snapshot, pick a random cube of size (lr/hr)_patchsize**3
        # If boxsize/patchsize is not integer, you won't fully exploit your snapshots
        # One batch will is filled with first by as many cubes as you can get from
        # a single snapshot. It could be better to fill batches with data from different snapshots.

        for bNum, snap_id in enumerate(snap_ids):
            # Select the patch position in the snapshot
            bIDX, bIDY, bIDZ = np.random.randint(0, nPatches, 3)

            file = fList[snap_id]

            with h5.File(file, 'r') as f:
                hr = f['HR']
                lr = f['LR']
                for cNum, channel in enumerate(self.channels):
                    patch_lr = np.array(lr[channel])
                    patch_hr = np.array(hr[channel])
                    # Take log(P) and log(rho)
                    if channel == 'rho':
                        patch_lr = np.log10(patch_lr)
                        patch_hr = np.log10(patch_hr)

                    # Data normalization... to be explored
                    mean = np.mean(patch_hr)
                    std = np.std(patch_hr)

                    patch_lr = (patch_lr-mean)/std
                    patch_hr = (patch_hr-mean)/std

                    patch_lr = patch_lr[bIDX*ps_lr:(bIDX+1)*ps_lr,
                                        bIDY*ps_lr:(bIDY+1)*ps_lr,
                                        bIDZ*ps_lr:(bIDZ+1)*ps_lr]
                    patch_hr = patch_hr[bIDX*ps_hr:(bIDX+1)*ps_hr,
                                        bIDY*ps_hr:(bIDY+1)*ps_hr,
                                        bIDZ*ps_hr:(bIDZ+1)*ps_hr]

                    lr_train[bNum,:,:,:,cNum] = patch_lr
                    hr_train[bNum,:,:,:,cNum] = patch_hr

        return lr_train, hr_train

    def loadSnapshot(self, idx):
        """
        Load a full box from a snapshot.
        """

        # Number of patches needed to fill the input box
        batch_size = 1
        hr_size = self.hr_boxsize
        lr_size = self.hr_boxsize // self.sr_factor

        lr_train = np.zeros([batch_size,lr_size,lr_size,lr_size,self.nChannels] )
        hr_train = np.zeros([batch_size,hr_size,hr_size,hr_size,self.nChannels] )

        for bNum in range(batch_size):
            file = self.fList_t[idx%self.nFiles_t]

            with h5.File(file, 'r') as f:
                hr = f['HR']
                lr = f['LR']
                for cNum, channel in enumerate(self.channels):
                    patch_lr = np.array(lr[channel])
                    patch_hr = np.array(hr[channel])
                    # Take log(P) and log(rho)
                    if channel == 'rho':
                        if np.min(patch_hr)<0:
                            print(file)
                        patch_lr = np.log10(patch_lr)
                        patch_hr = np.log10(patch_hr)
                        if np.sum(np.isnan(patch_hr))>0:
                            print(file)

                    # Data normalization... to be explored
                    mean = np.mean(patch_hr)
                    std = np.std(patch_hr)
                    if channel == 'rho':
                        print(mean, std)

                    patch_lr = (patch_lr-mean)/std
                    patch_hr = (patch_hr-mean)/std

                    lr_train[bNum,:,:,:,cNum] = patch_lr
                    hr_train[bNum,:,:,:,cNum] = patch_hr

        return lr_train, hr_train

    def loadRandomBatch_noise(self, batch_size):
        """
        For testing only: put some random numbers in the batch
        """
        ps_lr = self.hr_patchsize // self.sr_factor
        ps_hr = self.hr_patchsize
        lr_train = np.zeros([batch_size,ps_lr,ps_lr,ps_lr,self.nChannels] )
        hr_train = np.zeros([batch_size,ps_hr,ps_hr,ps_hr,self.nChannels] )
        nPatches = self.hr_boxsize // self.hr_patchsize
        x_ = np.linspace(-1., 1., ps_lr)
        x_lr, y_lr, z_lr = np.meshgrid(x_, x_, x_)
        x_ = np.linspace(-1., 1., ps_hr)
        x_hr, y_hr, z_hr = np.meshgrid(x_, x_, x_)
        for bNum in range(batch_size):
            for cNum, channel in enumerate(self.channels):
                patch = np.random.rand(ps_lr,ps_lr,ps_lr) * np.cos(2*np.pi*(x_lr**2 + y_lr**2 + z_lr**2)) * 10
                lr_train[bNum,:,:,:,cNum] = normalize(patch)
            for cNum, channel in enumerate(self.channels):
                patch = np.random.rand(ps_hr,ps_hr,ps_hr) * np.cos(2*np.pi*(x_hr**2 + y_hr**2 + z_hr**2)) + 10
                hr_train[bNum,:,:,:,cNum] = normalize(patch)
        return lr_train, hr_train
