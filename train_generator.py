import tensorflow as tf
from SRISMt import SRISMt
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
run_dir = '/home/cluster/hlasco/scratch/generator/'

print('Initializing Networks', flush=True)

gan = SRISMt(
        data_directory=data_dir,
        lr_patchsize=16,
        hr_patchsize=64,
        output_dir = run_dir,
        lRate_G = 1e-4,
        lRate_D = 1e-4,
        loss_weights={'pixel':1.0, 'TE':1.0,'MF':1.0,'ENS':1.0,'adversarial':0.01, },
        nChannels=4,
        bNorm=False,
        training_mode=True
    )
print('Training Generator', flush=True)
#gan.restart(gen_w=run_dir + 'SR-RRDB-G_4X.h5', dis_w=run_dir + 'SR-RRDB-D_4X.h5', epoch_0=1000)
gan.train_generator(batch_size=1, step_per_epoch=4, n_epochs=10000)
