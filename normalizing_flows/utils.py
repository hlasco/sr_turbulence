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
from flows.cglow.cond_affine_coupling import cond_coupling_nn_glow
from flows.cglow.cond_gaussianize import cond_gaussianize
from flows.glow.affine_coupling import coupling_nn_glow
from flows.cglow.affine_injector import injector_nn_glow
from models.flvm import FlowLVM

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

def get_kwargs(config, section, keys):
    config = config[section]
    kwargs = {k:int(v) for k,v in config.items() if k in keys}
    return kwargs

def get_rundir(config):
    base_path = config['paths']['run_dir']
    k1 = get_kwargs(config, 'flow', ['num_layers', 'depth', 'min_filters', 'max_filters']).values()
    k2 = get_kwargs(config, 'cond', ['cond_filters', 'cond_resblocks', 'cond_blocks']).values()
    ctype = config['training']['channel_type']
    run_dir = base_path + "/flow_{}_{}_{}_{}_cond_{}_{}_{}_{}/".format(*k1, *k2, ctype)
    return run_dir

def get_bInit(config):
    if int(config['training']['restart'])==0:
        return True
    rundir = get_rundir(config)
    ckpt_path = tf.train.latest_checkpoint(rundir)
    if ckpt_path != '':
        return False
    else:
        return True

def get_model(config, restart=False):
    dim = int(config['flow']['dim'])
    inpt_channels = get_inpt_channels(config)
    cond_channels = get_cond_channels(config)

    kwargs_nn = get_kwargs(config, section='flow',
         keys=['dim','min_filters', 'max_filters', 'num_blocks'])

    cond_coupling = cond_coupling_nn_glow(**kwargs_nn)
    injector = injector_nn_glow(**kwargs_nn)
    parametrizer = cond_gaussianize(**kwargs_nn)

    kwargs_flow = get_kwargs(config, section='flow', 
         keys=['upfactor', 'num_layers', 'depth'])

    kwargs_cond = get_kwargs(config, section='cond', 
         keys=['cond_channels', 'cond_filters', 'cond_resblocks', 'cond_blocks'])


    glow = Invert(CGlowFlowSR(**kwargs_flow, **kwargs_cond,
                              dim=dim, cond_channels=cond_channels,
                              cond_coupling_nn_ctor=cond_coupling,
                              injector_nn_ctor=injector,
                              parameterize_ctor=parametrizer))

    learning_rate = float(config['training']['learning_rate'])
    num_bins = int(config['training']['num_bins'])

    prior = tfp.distributions.Normal(loc=0.0, scale=1.0)
    opt_flow = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    opt_cond = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model = FlowLVM(glow, prior, dim=dim, num_bins=num_bins, input_channels=inpt_channels,
                    cond_channels=cond_channels, optimizer_flow=opt_flow, optimizer_cond=opt_cond)
    print("Model built with",model.param_count().numpy(),"parameters.", flush=True)
    model._init_checkpoint()

    if int(config['training']['restart']) > 0 or restart:
        rundir = get_rundir(config)
        print("Restoring from latest checkpoint in:", rundir)
        ckpt_path = tf.train.latest_checkpoint(rundir)
        print("Checkpoint found:", ckpt_path)
        model.checkpoint.restore(ckpt_path)

    return model

def get_dataset(config):
    dset_dir = config['paths']['dset_dir']
    turb_type = config['training']['turb_type']
    upfactor = int(config['flow']['upfactor'])
    hr_patch_size = int(config['training']['hr_patch_size'])
    hr_sim_size = int(config['training']['hr_sim_size'])
    channel_type = config['training']['channel_type']
    if turb_type == 'compressible':
        mach = list(map(int, config['training']['mach'].split(',')))
        if int(config['flow']['dim']) == 3:
            return get_dataset_compressible_3d(dset_dir, mach, hr_patch_size, hr_sim_size, upfactor, channel_type)
        else:
            # return get_dataset_compressible_2d(dset_dir, mach, hr_patch_size, hr_sim_size, upfactor, channel_type)
            raise ValueError("Compressible 2d dataset not implemented yet.")
    else:
        raise ValueError("Incompressible dataset not implemented yet.")

def get_dataset_compressible_3d(dset_dir, mach, hr_ps, hr_sim_size, upfactor, channel_type):

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
            mach = sigma_u / cs
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



        return ret_hr, ret_lr #, mach

    def readH5_wrapper(filename):
        hr, lr = tf.py_function(readH5, [filename], (tf.float32, tf.float32))
        return hr, lr

    sim_list = sorted(glob.glob(dset_dir + "/train/mach_*/"))
    mach_list = np.array([int(m[-3:-1]) for m in sim_list], dtype=int)
    for i in range(len(mach_list)):
        sim_list[i] = glob.glob(sim_list[i] + "*/processed_data/snapshot.h5")
        if sim_list[i] == []:
            mach_list.pop(i)
    try:
        sim_list.remove([])
    except:
        pass
    to_ignore = np.array([m not in mach for m in mach_list])
    to_ignore = np.where(to_ignore)

    for index in sorted(to_ignore, reverse=True):
        mach_list = np.delete(mach_list,index)
        sim_list = np.delete(sim_list,index)

    sim_list = list(itertools.chain(*sim_list))
    random.shuffle(sim_list)
    dataset = tf.data.Dataset.from_tensor_slices(sim_list)
    dataset = dataset.map(lambda x: readH5_wrapper(x))
    dataset = dataset.repeat()
    return dataset, num_patches**3
