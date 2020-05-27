import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Lambda
import h5py as h5
import matplotlib.pyplot as plt
import os, io, glob, sys

#import horovod.tensorflow.keras as hvd
import numpy as np

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
                 LR_directory,      # Directory containing the High-res snapshots
                 HR_directory,      # Directory containing the High-res snapshots
                 HR_boxsize=128,    # High-res full box size
                 sr_factor=4,       # Super-resolution factor
                 HR_patchsize=64,   # High-res patch size
                 nChannels=5,       # Number of channels (u,v,w,rho,P)
                 ):

        self.HR_directory = HR_directory
        self.LR_directory = LR_directory
        self.HR_boxsize   = HR_boxsize
        self.sr_factor    = sr_factor
        self.HR_patchsize = HR_patchsize
        self.nChannels    = nChannels

        # Just to read my hdf5 snapshots
        self.channels = ['velocity_x','velocity_y','velocity_z','density','pressure']

        # Make sure that your file names can be sorted
        self.HR_fList = sorted(glob.glob(HR_directory+'/*.h5'))
        self.HR_nFile = len(self.HR_fList)

        self.LR_fList = sorted(glob.glob(LR_directory+'/*.h5'))
        self.nFiles = len(self.LR_fList)

    def loadRandomBatch(self, batch_size):
        """
        Load a batch of data for training.
        """
        
        # High-res and Low-res files should go by pair
        if self.nFiles is not len(self.HR_fList):
            raise ValueError('HR and LR directories should contain the same number of snapshots \
                              LR_nFile={}, HR_nFile={}'.format(self.nFiles, len(self.HR_fList)))

        ps_lr = self.HR_patchsize // self.sr_factor
        ps_hr = self.HR_patchsize

        LR_train = np.zeros([batch_size,ps_lr,ps_lr,ps_lr,self.nChannels] )
        HR_train = np.zeros([batch_size,ps_hr,ps_hr,ps_hr,self.nChannels] )
        
        # Number of snapshots needed to fill the batch
        nPatches = self.HR_boxsize // self.HR_patchsize
        
        snap_ids = np.random.randint(0, self.nFiles, batch_size) # Here I can split train/test :)

        # For each random snapshot, pick a random cube of size (LR/HR)_patchsize**3
        # If boxsize/patchsize is not integer, you won't fully exploit your snapshots
        # One batch will is filled with first by as many cubes as you can get from
        # a single snapshot. It would be nice to fill batches with independant patches.
        
        
        for bNum, snap_id in enumerate(snap_ids):
            # Select the patch position in the snapshot
            bIDX, bIDY, bIDZ = np.random.randint(0, nPatches, 3)
            
            LR_file = self.LR_fList[snap_id]
            HR_file = self.HR_fList[snap_id]

            # Deal with Low-res data
            with h5.File(LR_file, 'r') as f:
                gas = f['gas']
                info = f['info']
                dims = info.attrs['dims']
                for cNum, channel in enumerate(self.channels):
                    patch = np.array(gas[channel]).reshape(dims)
                    patch = patch[bIDX*ps_lr:(bIDX+1)*ps_lr,
                                  bIDY*ps_lr:(bIDY+1)*ps_lr,
                                  bIDZ*ps_lr:(bIDZ+1)*ps_lr]
                    # Take log(P) and log(rho)
                    if channel in ['density', 'pressure']:
                        patch = np.log(patch)

                    # Data normalization... to be explored
                    LR_train[bNum,:,:,:,cNum] = normalize(patch)

            # Deal with High-res data. Similar
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
                    HR_train[bNum,:,:,:,cNum] = normalize(patch)

        return LR_train, HR_train
        
    def loadRandomBatch_noise(self, batch_size):
        """
        For testing only: put some random numbers in the batch
        """
        ps_lr = self.HR_patchsize // self.sr_factor
        ps_hr = self.HR_patchsize
        LR_train = np.zeros([batch_size,ps_lr,ps_lr,ps_lr,self.nChannels] )
        HR_train = np.zeros([batch_size,ps_hr,ps_hr,ps_hr,self.nChannels] )
        nPatches = self.HR_boxsize // self.HR_patchsize
        x_ = np.linspace(-1., 1., ps_lr)
        x_lr, y_lr, z_lr = np.meshgrid(x_, x_, x_)
        x_ = np.linspace(-1., 1., ps_hr)
        x_hr, y_hr, z_hr = np.meshgrid(x_, x_, x_)
        for bNum in range(batch_size):
            for cNum, channel in enumerate(self.channels):
                patch = np.random.rand(ps_lr,ps_lr,ps_lr) * np.cos(2*np.pi*(x_lr**2 + y_lr**2 + z_lr**2)) * 10
                LR_train[bNum,:,:,:,cNum] = normalize(patch)
            for cNum, channel in enumerate(self.channels):
                patch = np.random.rand(ps_hr,ps_hr,ps_hr) * np.cos(2*np.pi*(x_hr**2 + y_hr**2 + z_hr**2)) + 10
                HR_train[bNum,:,:,:,cNum] = normalize(patch)
        return LR_train, HR_train
