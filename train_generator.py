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


data_dir = '/home/cluster/hlasco/scratch/boxicgen/'

print('Initializing Networks', flush=True)

gan = PISRT_GAN(
        data_directory=data_dir,
        lr_patchsize=16,
        hr_patchsize=64,
        output_dir = '/home/cluster/hlasco/scratch/gan/generator',
        lRate_G = 1e-4,
        lRate_D = 1e-7,
        nChannels=4,
        bNorm=False,
        training_mode=True
    )
print('Training Generator', flush=True)
#gan.restart('/home/cluster/hlasco/scratch/gan/generator_new/SR-RRDB-G_4X.h5', 4880)
gan.train_generator(batch_size=8, step_per_epoch=8, n_epochs=1000)
