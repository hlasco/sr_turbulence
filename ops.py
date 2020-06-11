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

def normalize(patch):
    """
    Normalize a patch of fluid quantities.
    """
    mu = np.mean(patch)
    std = np.std(patch)
    return (patch-mu)/std

class DataLoader():
    """
    Tool for loading snapshots from a directory
    """

    def __init__(self,
                 lr_directory,      # Directory containing the High-res snapshots
                 hr_directory,      # Directory containing the High-res snapshots
                 hr_boxsize=128,    # High-res full box size
                 sr_factor=4,       # Super-resolution factor
                 hr_patchsize=64,   # High-res patch size
                 nChannels=5,       # Number of channels (u,v,w,rho,P)
                 ):

        self.hr_directory = hr_directory
        self.lr_directory = lr_directory
        self.hr_boxsize   = hr_boxsize
        self.sr_factor    = sr_factor
        self.hr_patchsize = hr_patchsize
        self.nChannels    = nChannels

        # Just to read my hdf5 snapshots
        self.channels = ['velocity_x','velocity_y','velocity_z','density','pressure']

        # Make sure that your file names can be sorted
        self.hr_fList = sorted(glob.glob(hr_directory+'/*.h5'))[3:]
        self.hr_nFile = len(self.hr_fList)

        self.lr_fList = sorted(glob.glob(lr_directory+'/*.h5'))[3:]
        self.nFiles = len(self.lr_fList)

    def loadRandomBatch(self, batch_size):
        """
        Load a batch of data for training.
        """

        # High-res and Low-res files should go by pair
        if self.nFiles is not len(self.hr_fList):
            raise ValueError('hr and lr directories should contain the same number of snapshots \
                              lr_nFile={}, hr_nFile={}'.format(self.nFiles, len(self.hr_fList)))

        ps_lr = self.hr_patchsize // self.sr_factor
        ps_hr = self.hr_patchsize

        lr_train = np.zeros([batch_size,ps_lr,ps_lr,ps_lr,self.nChannels] )
        hr_train = np.zeros([batch_size,ps_hr,ps_hr,ps_hr,self.nChannels] )

        # Number of snapshots needed to fill the batch
        nPatches = self.hr_boxsize // self.hr_patchsize

        snap_ids = np.random.randint(0, self.nFiles, batch_size) # Here I can split train/test :)

        # For each random snapshot, pick a random cube of size (lr/hr)_patchsize**3
        # If boxsize/patchsize is not integer, you won't fully exploit your snapshots
        # One batch will is filled with first by as many cubes as you can get from
        # a single snapshot. It would be nice to fill batches with independant patches.


        for bNum, snap_id in enumerate(snap_ids):
            # Select the patch position in the snapshot
            bIDX, bIDY, bIDZ = np.random.randint(0, nPatches, 3)

            lr_file = self.lr_fList[snap_id]
            hr_file = self.hr_fList[snap_id]

            # Deal with Low-res data
            with h5.File(lr_file, 'r') as f:
                gas = f['gas']
                info = f['info']
                dims = info.attrs['dims']
                for cNum, channel in enumerate(self.channels):
                    patch = np.array(gas[channel]).reshape(dims)
                    # Take log(P) and log(rho)
                    if channel in ['density', 'pressure']:
                        patch = np.log(patch)

                    # Data normalization... to be explored
                    patch = normalize(patch)
                    patch = patch[bIDX*ps_lr:(bIDX+1)*ps_lr,
                                  bIDY*ps_lr:(bIDY+1)*ps_lr,
                                  bIDZ*ps_lr:(bIDZ+1)*ps_lr]

                    lr_train[bNum,:,:,:,cNum] = patch

            # Deal with High-res data. Similar
            with h5.File(hr_file, 'r') as f:
                gas = f['gas']
                info = f['info']
                dims = info.attrs['dims']
                for cNum, channel in enumerate(self.channels):
                    patch = np.array(gas[channel]).reshape(dims)
                    if channel in ['density', 'pressure']:
                        patch = np.log(patch)
                    patch = normalize(patch)
                    patch = patch[bIDX*ps_hr:(bIDX+1)*ps_hr,
                                  bIDY*ps_hr:(bIDY+1)*ps_hr,
                                  bIDZ*ps_hr:(bIDZ+1)*ps_hr]

                    hr_train[bNum,:,:,:,cNum] = patch

        return lr_train, hr_train

    def loadSnapshot(self, idx):
        """
        Load a batch of data for training.
        """

        # High-res and Low-res files should go by pair
        if self.nFiles is not len(self.hr_fList):
            raise ValueError('hr and lr directories should contain the same number of snapshots \
                              lr_nFile={}, hr_nFile={}'.format(self.nFiles, len(self.hr_fList)))

        ps_lr = self.hr_patchsize // self.sr_factor
        ps_hr = self.hr_patchsize

        # Number of patchss needed to fill the input box
        batch_size = (self.hr_boxsize // self.hr_patchsize)**3
        size = self.hr_boxsize // self.hr_patchsize
        lr_train = np.zeros([batch_size,ps_lr,ps_lr,ps_lr,self.nChannels] )
        hr_train = np.zeros([batch_size,ps_hr,ps_hr,ps_hr,self.nChannels] )
        snap_id = idx

        for index in range(batch_size):
            # Select the patch position in the snapshot

            bIDX = index // (size * size)
            bIDY = (index // size) % size
            bIDZ = index % size

            lr_file = self.lr_fList[snap_id]
            hr_file = self.hr_fList[snap_id]

            # Deal with Low-res data
            with h5.File(lr_file, 'r') as f:
                gas = f['gas']
                info = f['info']
                dims = info.attrs['dims']
                for cNum, channel in enumerate(self.channels):
                    patch = np.array(gas[channel]).reshape(dims)
                    # Take log(P) and log(rho)
                    if channel in ['density', 'pressure']:
                        patch = np.log(patch)
                    patch = normalize(patch)
                    patch = patch[bIDX*ps_lr:(bIDX+1)*ps_lr,
                                  bIDY*ps_lr:(bIDY+1)*ps_lr,
                                  bIDZ*ps_lr:(bIDZ+1)*ps_lr]


                    # Data normalization... to be explored
                    lr_train[index,:,:,:,cNum] = patch

            # Deal with High-res data. Similar
            with h5.File(hr_file, 'r') as f:
                gas = f['gas']
                info = f['info']
                dims = info.attrs['dims']
                for cNum, channel in enumerate(self.channels):
                    patch = np.array(gas[channel]).reshape(dims)
                    if channel in ['density', 'pressure']:
                        patch = np.log(patch)
                    patch = normalize(patch)
                    patch = patch[bIDX*ps_hr:(bIDX+1)*ps_hr,
                                  bIDY*ps_hr:(bIDY+1)*ps_hr,
                                  bIDZ*ps_hr:(bIDZ+1)*ps_hr]

                    hr_train[index,:,:,:,cNum] = patch

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
