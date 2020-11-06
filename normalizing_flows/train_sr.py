#!/usr/bin/env python3
import os
import tensorflow as tf

nGPUs = len(tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", nGPUs, flush=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)
import utils as u
import argparse
from shutil import copyfile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Train flow model for super-resolution")
    parser.add_argument('--config_file', dest='config_file', help="Configuration file.")
    parser.set_defaults(config_file='config_sr.ini')
    args = parser.parse_args()

    config = u.MyConfigParser()
    config.read(args.config_file)

    print('Using configuration file :{}'.format(args.config_file), flush=True)
    with open(args.config_file, 'r') as f:
        print(f.read(), flush=True)

    rundir = u.get_rundir(config)

    os.makedirs(rundir, exist_ok=True)
    copyfile(args.config_file, rundir+'/config.ini')

    model = u.get_model(config)
    print("Build model with",model.param_count(),"parameters.", flush=True)
    bInit = u.get_bInit(config)

    ckpt_freq = config.getint('training', 'ckpt_freq')
    nsteps_tot = config.getint( 'training',  'nsteps')
    bs = config.getint('training', 'batch_size')

    dset, dset_bs = u.get_dataset(config)

    n_ckpt = nsteps_tot // ckpt_freq
    for i in range(n_ckpt):
        print('\nEpoch {}:'.format(i), flush=True)
        model.train(dset, bs=bs, dset_bs=dset_bs, steps_per_epoch=ckpt_freq,
                   num_epochs=1, conditional=True, init=bInit)
        print('\nSaving model',  flush=True)
        model.save(rundir + 'model')
