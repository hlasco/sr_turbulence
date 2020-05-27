import tensorflow as tf
from PISRT_GAN import PISRT_GAN
import os, sys

#import horovod.tensorflow.keras as hvd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

#hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
#gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)
#if gpus:
#    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


lr_dir = '/home/cluster/hlasco/scratch/ramses3d/pbox_n50_noGrav/post_processing/covering_grids_5/'
hr_dir = '/home/cluster/hlasco/scratch/ramses3d/pbox_n50_noGrav/post_processing/covering_grids_7/'

print('Initializing Networks', flush=True)

gan = PISRT_GAN(
        LR_directory=lr_dir,
        HR_directory=hr_dir,
        LR_patchsize=16,
        HR_patchsize=64,
        output_dir = './rundir',
        lRate_G = 1e-4,
        lRage_D = 1e-7,
        nChannels=5,
        training_mode=True
    )
            
gan.train_generator(batch_size=8, step_per_epoch=4, n_epochs=100)
