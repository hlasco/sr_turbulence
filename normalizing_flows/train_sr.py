#!/usr/bin/env python3
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)



import utils as u
import argparse
from shutil import copyfile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Train flow model for super-resolution")
    parser.add_argument('--config_file', dest='config_file', help="Configuration file.")
    parser.set_defaults(config_file='launch/settings/config_sr4.ini')
    args = parser.parse_args()

    strategy = tf.distribute.get_strategy()

    config = u.MyConfigParser()
    config.read(args.config_file)

    print('Using configuration file :{}'.format(args.config_file), flush=True)
    with open(args.config_file, 'r') as f:
        print(f.read(), flush=True)

    rundir = u.get_rundir(config)

    os.makedirs(rundir, exist_ok=True)
    copyfile(args.config_file, rundir+'/config.ini')
    with strategy.scope():
        model = u.get_model(config)
    ckpt_num = u.get_ckpt_num(config)
    bInit = u.get_bInit(config)

    ckpt_freq = config.getint('training', 'ckpt_freq')
    nsteps_tot = config.getint( 'training',  'nsteps')
    batch_size = config.getint('training', 'batch_size')

    dset = u.get_dataset(config)

    n_ckpt = nsteps_tot // ckpt_freq
    print('\nStarting training:', flush=True)
    model.train(dset, batch_size=batch_size, steps_per_epoch=ckpt_freq, num_epochs=n_ckpt,
                epoch_0=ckpt_num, conditional=True, init=bInit)
