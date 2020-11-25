import sys, os, glob
import h5py
import random
import itertools
sys.path += ['.']

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from flows.cglow import CGlowFlowSR
from flows.invert import Invert
from flows.cglow.cond_gaussianize import CondGaussianize
from flows.glow.gaussianize import Gaussianize
from flows.networks.cond_nn import cond_nn
from flows.networks.coupling_nn import coupling_nn
from models.flvm import FlowLVM
from models.utils import NormalPrior

from configparser import RawConfigParser
from configparser import ConfigParser, NoOptionError

def mach_to_loc(mach):
    m = 0.40409965475306253
    M = 9.773589951034829
    loc = 2*((mach - m) / (M+m) - 0.5)
    return loc

class MyConfigParser(RawConfigParser):
    def get(self, section, option):
        try:
            return RawConfigParser.get(self, section, option)
        except NoOptionError:
            return None
    def getint(self, section, option):
        try:
            return int(RawConfigParser.get(self, section, option))
        except NoOptionError:
            return None
    def getfloat(self, section, option):
        try:
            return float(RawConfigParser.get(self, section, option))
        except NoOptionError:
            return None

def get_inpt_channels(config):
    config = config['training']

    assert config['turb_type'] in ['compressible', 'incompressible'], \
           "Invalid turbulence type: {}. Use (compressible/incompressible)".format(config['turb_type'])

    assert config['channel_type'] in ['s', 'vel', 'all'], \
           "Invalid channel type: {}. Use (s/vel/all)".format(config['channel_type'])

    if config['turb_type'] == 'incompressible':
        assert config['channel_type'] in ['vel', 'all'], \
               "Invalid channel_type for {} turbulence. Use (vel/all).".format(config['turb_type'])
        return 3

    if config['channel_type'] == 's':
        return 1
    elif config['channel_type'] == 'vel':
        return 3
    elif config['channel_type'] == 'all':
        return 4

def get_cond_channels(config):
    config = config['training']

    if config['turb_type'] == 'incompressible':
        return 3
    else:
        return 4

def get_kwargs(config, keys):
    kwargs = {}
    for section in config.sections():
        conf = config[section]
        for k,v in conf.items():
            if k in keys:
                kwargs[k] = int(v)
    return kwargs

def get_rundir(config):
    base_path = config['paths']['run_dir']
    k = get_kwargs(config, ['num_layers', 'depth', 'min_filters', 'max_filters', \
                            'cond_filters', 'cond_resblocks', 'cond_blocks']).values()
    upfactor = int(2**config.getint('flow','upfactor'))
    ctype = config['training']['channel_type']
    run_dir = base_path + "/flowX{}_{}_{}_{}_{}_cond_{}_{}_{}_{}/".format(upfactor, *k, ctype)
    #run_dir = base_path + "/flow_{}_{}_{}_{}_cond_{}_{}_{}_{}/".format(*k1, *k2, ctype)
    return run_dir

def get_bInit(config):
    if config.getint('training','restart')==0:
        return True
    rundir = get_rundir(config)
    ckpt_path = tf.train.latest_checkpoint(rundir)
    if ckpt_path != '':
        return False
    else:
        return True

def get_model(config, restart=False):
    dim = config.getint('flow','dim')
    rundir = get_rundir(config)
    inpt_channels = get_inpt_channels(config)
    cond_channels = get_cond_channels(config)

    kwargs_nn = get_kwargs(config,
         keys=['dim','min_filters', 'max_filters', 'num_blocks'])
    kwargs_flow = get_kwargs(config,
         keys=['upfactor', 'num_layers', 'depth'])
    kwargs_cond = get_kwargs(config,
         keys=['dim', 'cond_filters', 'cond_resblocks', 'cond_blocks'])
    kwargs_cond['cond_channels'] = cond_channels

    print("Building model:")

    coupling_ctor = coupling_nn(**kwargs_nn)
    cond_ctor = cond_nn(**kwargs_cond)
    #parametrizer = CondGaussianize(**kwargs_nn)
    parametrizer = Gaussianize(**kwargs_nn)

    glow = Invert(CGlowFlowSR(**kwargs_flow, **kwargs_cond,
                              coupling_ctor=coupling_ctor,
                              cond_ctor=cond_ctor,
                              parameterize_ctor=parametrizer))

    learning_rate = float(config.getfloat('training','learning_rate'))
    num_bins = config.getint('training', 'num_bins')

    prior = NormalPrior(loc=0.0, scale=1.0)
    opt_flow = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    opt_cond = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model = FlowLVM(glow, prior, dim=dim, num_bins=num_bins, input_channels=inpt_channels,
                    cond_channels=cond_channels, optimizer_flow=opt_flow, optimizer_cond=opt_cond, rundir=rundir)
    print("Model built with",model.param_count().numpy(),"parameters.", flush=True)
    model._init_checkpoint()

    if config.getint('training','restart') > 0 or restart:
        rundir = get_rundir(config)
        print("Trying to restart from latest checkpoint in:", rundir)
        ckpt_path = tf.train.latest_checkpoint(rundir)
        print("Checkpoint found:", ckpt_path)
        model.checkpoint.restore(ckpt_path)

    return model

def get_ckpt_num(config):
    rundir = get_rundir(config)
    ckpt_path = tf.train.latest_checkpoint(rundir)
    if ckpt_path != None:
        ckpt_num = int(ckpt_path.split('-')[-1])
    else:
        ckpt_num = 0
    return ckpt_num

def get_dataset(config):

    dset_dir = config.get('paths','dset_dir')
    turb_type = config.get('training','turb_type')
    upfactor = config.getint('flow','upfactor')
    hr_patch_size = config.getint('training','hr_patch_size')
    hr_sim_size = config.getint('training','hr_sim_size')
    channel_type = config.get('training','channel_type')
    batch_size = config.getint('training', 'batch_size')
    if turb_type == 'compressible':
        mach = list(map(int, config['training']['mach'].split(',')))
        if config.getint('flow','dim')== 3:
            dset = get_dataset_compressible_3d(dset_dir, mach, hr_patch_size, hr_sim_size, upfactor, channel_type, batch_size)
            return dset
        else:
            dset = get_dataset_compressible_2d(dset_dir, mach, hr_patch_size, hr_sim_size, upfactor, channel_type)
            dset = dset.batch(batch_size)
            return dset
    else:
        raise ValueError("Incompressible dataset not implemented yet.")

def get_dataset_compressible_3d(dset_dir, mach, hr_ps, hr_sim_size, upfactor, channel_type, batch_size):

    channels_lr = ['ux', 'uy', 'uz', 's']

    if channel_type == 'vel':
        channels_hr = ['ux', 'uy', 'uz']
    elif channel_type == 's':
        channels_hr = ['s']
    else:
        channels_hr = ['ux', 'uy', 'uz', 's']



    upfactor = int(2**upfactor)
    lr_ps = hr_ps // upfactor
    num_patches = hr_sim_size // hr_ps

    def readH5(f):
        f = f.numpy()
        ret_lr = np.zeros([num_patches**3,lr_ps,lr_ps,lr_ps,len(channels_lr)] )
        ret_hr = np.zeros([num_patches**3,hr_ps,hr_ps,hr_ps,len(channels_hr)] )
        with h5py.File(f, 'r') as fi:
            lr_key = 'FILT{}/'.format(upfactor)
            att = dict(fi[lr_key].attrs)
            # Forgot to save mach number...
            cs = 0.18656300264905473
            sigma_u = (1./3 * (att['ux_std2'] + att['uy_std2'] + att['uz_std2']))**.5
            mach = np.float32(sigma_u / cs)
            for numc, c in enumerate(channels_lr):

                lr = np.array(fi[lr_key+c], dtype=np.float32)

                mean = att['{}_mean'.format(c)]
                sdev = att['{}_std2'.format(c)]**.5

                if c in channels_hr:
                    hr = np.array(fi['HR/'+c], dtype=np.float32)
                    hr = (hr - mean)/sdev
                lr = (lr - mean)/sdev

                x = np.arange(0,num_patches,1)
                ii,jj,kk = np.meshgrid(x,x,x)

                for i,j,k in zip(ii.flat, jj.flat, kk.flat):
                    num_patch = i + num_patches * (j + num_patches * k)
                    if c in channels_hr:
                        numc_hr = channels_hr.index(c)
                        ret_hr[num_patch, :,:,:,numc_hr] = hr[i*hr_ps:(i+1)*hr_ps,
                                                           j*hr_ps:(j+1)*hr_ps,
                                                           k*hr_ps:(k+1)*hr_ps]

                    ret_lr[num_patch, :,:,:,numc] = lr[i*lr_ps:(i+1)*lr_ps,
                                                       j*lr_ps:(j+1)*lr_ps,
                                                       k*lr_ps:(k+1)*lr_ps]

        return ret_hr, ret_lr, mach

    def readH5_wrapper(filename):
        hr, lr , mach= tf.py_function(readH5, [filename], (tf.float32, tf.float32, tf.float32))
        mach = mach*tf.ones(num_patches**3)
        ret = {'x': hr, 'y':lr, 'mach':mach}
        return ret

    sim_list = [dset_dir + "/train/mach_{}/".format(str(m).zfill(2)) for m in mach]

    for i in range(len(sim_list)):
        sim_list[i] = glob.glob(sim_list[i] + "*/processed_data/snapshots.h5")
    try:
        sim_list.remove([])
    except:
        pass

    sim_list = list(itertools.chain(*sim_list))
    dataset = tf.data.Dataset.from_tensor_slices(sim_list)
    dataset = dataset.map(lambda x: readH5_wrapper(x))
    dataset = dataset.unbatch()
    dataset = dataset.shuffle(num_patches**3*len(sim_list)).repeat()
    dataset = dataset.batch(batch_size)
    strategy = tf.distribute.get_strategy()
    dataset = strategy.experimental_distribute_dataset(dataset)
    return dataset

def get_dataset_compressible_2d(dset_dir, mach, hr_ps, hr_sim_size, upfactor, channel_type):

    channels_lr = ['ux', 'uy', 'uz', 's']

    if channel_type == 'vel':
        channels_hr = ['ux', 'uy', 'uz']
    elif channel_type == 's':
        channels_hr = ['s']
    else:
        channels_hr = ['ux', 'uy', 'uz', 's']

    upfactor = int(2**upfactor)
    lr_ps = hr_ps // upfactor
    num_patches = hr_sim_size // hr_ps
    lr_sim_size = hr_sim_size//upfactor

    def readH5(f):
        f = f.numpy()
        ret_lr = np.zeros([num_patches**2*lr_sim_size,lr_ps,lr_ps,len(channels_lr)] )
        ret_hr = np.zeros([num_patches**2*lr_sim_size,hr_ps,hr_ps,len(channels_hr)] )
        with h5py.File(f, 'r') as fi:
            lr_key = 'FILT{}/'.format(upfactor)
            att = dict(fi[lr_key].attrs)
            # Forgot to save mach number...
            cs = 0.18656300264905473
            sigma_u = (1./3 * (att['ux_std2'] + att['uy_std2'] + att['uz_std2']))**.5
            mach = np.float32(sigma_u / cs)
            for numc, c in enumerate(channels_lr):

                lr = np.array(fi[lr_key+c], dtype=np.float32)

                mean = att['{}_mean'.format(c)]
                sdev = att['{}_std2'.format(c)]**.5

                if c in channels_hr:
                    hr = np.array(fi['HR/'+c], dtype=np.float32)
                    hr = (hr - mean)/sdev
                lr = (lr - mean)/sdev

                x = np.arange(0,num_patches,1)
                z = np.arange(0,lr_sim_size,1)
                ii,jj,kk = np.meshgrid(x,x,z)

                for i,j,k in zip(ii.flat, jj.flat, kk.flat):
                    num_patch = k + lr_sim_size * (j + num_patches * i)
                    if c in channels_hr:
                        numc_hr = channels_hr.index(c)
                        ret_hr[num_patch, :,:,numc_hr] = hr[i*hr_ps:(i+1)*hr_ps,
                                                            j*hr_ps:(j+1)*hr_ps,
                                                            k*upfactor]

                    ret_lr[num_patch, :,:,numc] = lr[i*lr_ps:(i+1)*lr_ps,
                                                     j*lr_ps:(j+1)*lr_ps,
                                                     k]

        return ret_hr, ret_lr

    def readH5_wrapper(filename):
        hr, lr = tf.py_function(readH5, [filename], (tf.float32, tf.float32))
        ret = {'x': hr, 'y':lr}
        return ret

    sim_list = [dset_dir + "/train/mach_{}/".format(str(m).zfill(2)) for m in mach]

    for i in range(len(sim_list)):
        sim_list[i] = glob.glob(sim_list[i] + "*/processed_data/snapshot.h5")
    try:
        sim_list.remove([])
    except:
        pass

    sim_list = list(itertools.chain(*sim_list))
    dataset = tf.data.Dataset.from_tensor_slices(sim_list)
    dataset = dataset.map(lambda x: readH5_wrapper(x))
    dataset = dataset.unbatch()
    dataset = dataset.shuffle(num_patches**2*len(sim_list)).repeat()
    dataset = dataset.batch(batch_size)
    strategy = tf.distribute.get_strategy()
    dataset = strategy.experimental_distribute_dataset(dataset)
    return dataset
