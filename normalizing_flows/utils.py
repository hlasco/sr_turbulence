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

def get_num_channels(config, where):
    assert where in ['inputs', 'targets']
    data_type = config['training']['data_type']                     
    assert data_type in ['compressible', 'incompressible', 'dmh'], \
           "Invalid turbulence type: {}. Use (compressible/incompressible/dmh)".format(data_type)

    config_ = config[f'{data_type}-dataset']


    if data_type == 'compressible':
        assert config_[where] in ['s', 'vel', 'all'], \
           "Invalid {} channel type: {}. Use (s/vel/all)".format(where, config_[where])
        if config_['target'] == 's':
            return 1
        elif config_[where] == 'vel':
            return 3
        elif config_[where] == 'all':
            return 4

    if data_type == 'incompressible':
        assert config_['targets'] in ['vx', 'vy', 'vz', 'all'], \
               "Invalid {} channel_type for {} dataset. Use (vx/vy/vz/all).".format(where, data_type)
        return 3

    if data_type == 'dmh':
        assert config_['targets'] in ['dmh', 'gas'], \
               "Invalid {} channel_type for {} dataset. Use (dmh/gas).".format(where, data_type)
        if where=='inputs':
            return 2
        else:
            return 1

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
    k = get_kwargs(config, ['num_layers', 'depth', 'min_filters', 'max_filters', 'num_blocks', \
                            'cond_filters', 'cond_resblocks', 'cond_blocks']).values()
    kernel_size = config.getint('flow','kernel_size')
    upfactor = int(2**config.getint('flow','upfactor'))
    
    data_type = config['training']['data_type']
    config_dataset = config[f'{data_type}-dataset']
    targets = config_dataset['targets']
    inputs = config_dataset['inputs']

    run_dir = base_path + "/flow_X{}_{}_{}_{}_{}_{}_cond_{}_{}_{}_data_{}_{}/".format(upfactor, *k, inputs, targets)
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

def get_model(config, restart=False, ckpt_num=0):
    dim = config.getint('flow','dim')
    upfactor = config.getint('flow','upfactor')
    num_layers = config.getint('flow','num_layers')
    rundir = get_rundir(config)
                     
    channels = get_num_channels(config, 'targets')
    cond_channels = get_num_channels(config, 'inputs')

    kwargs_nn = get_kwargs(config,
         keys=['dim','min_filters', 'max_filters', 'num_blocks', 'kernel_size'])
    kwargs_flow = get_kwargs(config,
         keys=['upfactor', 'num_layers', 'depth'])
    kwargs_cond = get_kwargs(config,
         keys=['dim', 'cond_filters', 'cond_resblocks', 'cond_blocks', 'kernel_size'])
    kwargs_cond['cond_channels'] = cond_channels

    print("Building model:")

    coupling_ctor = coupling_nn(**kwargs_nn)
    cond_ctor = cond_nn(upfactor=upfactor, num_layers=num_layers, **kwargs_cond)
    parametrizer = CondGaussianize(**kwargs_nn)

    glow = Invert(CGlowFlowSR(**kwargs_flow, **kwargs_cond,
                              coupling_ctor=coupling_ctor,
                              cond_ctor=cond_ctor,
                              parameterize_ctor=parametrizer))

    learning_rate = float(config.getfloat('training','learning_rate'))
    num_bins = config.getint('training', 'num_bins')

    prior = NormalPrior(loc=0.0, scale=1.0)
    model = FlowLVM(glow, prior, dim=dim, num_bins=num_bins, input_channels=channels,
                    cond_channels=cond_channels, learning_rate=learning_rate, rundir=rundir)
    print("Model built with",model.param_count().numpy(),"parameters.", flush=True)
    model._init_checkpoint()

    if config.getint('training','restart') > 0 or restart:
        rundir = get_rundir(config)
        if ckpt_num>0:
            ckpt_path = rundir + "model-{}".format(ckpt_num)
        else:
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

def get_dataset(config, strategy=tf.distribute.get_strategy()):

    
    upfactor = config.getint('flow','upfactor')
    data_type = config.get('training','data_type')
    batch_size = config.getint('training', 'batch_size')
    
    dset_section = f'{data_type}-dataset'

    dset_dir = config.get(dset_section,'dset_dir')
    inputs = config.get(dset_section,'inputs')
    targets = config.get(dset_section,'targets')

    if data_type == 'compressible':
        mach = list(map(int, config[dset_section]['mach'].split(',')))
        hr_patch_size = config.getint(dset_section,'hr_patch_size')
        hr_sim_size = config.getint(dset_section,'hr_sim_size')
        lr_type = config.get(dset_section, 'lr_type')
        if config.getint('flow','dim')== 3:
            dset = get_dataset_compressible_3d(dset_dir, mach, hr_patch_size, hr_sim_size, upfactor, lr_type, targets, batch_size, strategy)
            return dset
        else:
            dset = get_dataset_compressible_2d(dset_dir, mach, hr_patch_size, hr_sim_size, upfactor, lr_type, channel_type, batch_size, strategy)
            return dset

    elif data_type == 'dmh':
        assert config.getint('flow','dim')== 2, \
               "Invalid dim for {} turbulence. Use (2).".format(config['data_type'])
        redshifts = list(map(int, config[dset_section]['redshifts'].split(',')))
        tilesize = config.getint(dset_section,'tilesize')
        dset = get_dataset_dmh(dset_dir, redshifts, tilesize, inputs, targets, batch_size, strategy)
        return dset
    elif data_type == 'incompressible':
        raise ValueError("Incompressible dataset not implemented yet.")

    else:
        raise ValueError("Dataset not implemented yet.")

def get_dataset_compressible_3d(dset_dir, mach, hr_ps, hr_sim_size, upfactor, lr_type, channel_type, batch_size, strategy):

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

    if lr_type=='filt':
        lr_key = 'FILT{}/'.format(upfactor)
    elif lr_type=='sim':
        assert upfactor==4
        lr_key = 'LR/'

    hr_key='HR/'

    def readH5(f):
        f = f.numpy()
        ret_lr = np.zeros([num_patches**3,lr_ps,lr_ps,lr_ps,len(channels_lr)] )
        ret_hr = np.zeros([num_patches**3,hr_ps,hr_ps,hr_ps,len(channels_hr)] )
        with h5py.File(f, 'r') as fi:
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
                    hr = np.array(fi[hr_key+c], dtype=np.float32)
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
        sim_list[i] = glob.glob(sim_list[i] + "*/processed_data/snapshot.h5")
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
    dataset = strategy.experimental_distribute_dataset(dataset)
    return dataset

def get_dataset_compressible_2d(dset_dir, mach, hr_ps, hr_sim_size, upfactor, lr_type, channel_type, batch_size, strategy):

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

    if lr_type=='filt':
        lr_key = 'FILT{}/'.format(upfactor)
    elif lr_type=='sim':
        assert upfactor==4
        lr_key = 'LR/'

    def readH5(f):
        f = f.numpy()
        ret_lr = np.zeros([num_patches**2*lr_sim_size,lr_ps,lr_ps,len(channels_lr)])
        ret_hr = np.zeros([num_patches**2*lr_sim_size,hr_ps,hr_ps,len(channels_hr)])
        with h5py.File(f, 'r') as fi:
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
                    slice_lr = lr[:,:,k]
                    ret_lr[num_patch, :,:,numc] = slice_lr[i*lr_ps:(i+1)*lr_ps,
                                                           j*lr_ps:(j+1)*lr_ps]
                    
                    if c in channels_hr:
                        numc_hr = channels_hr.index(c)
                        slice_hr = hr[:,:,k*upfactor]
                        ret_hr[num_patch, :,:,numc_hr] = slice_hr[i*hr_ps:(i+1)*hr_ps,
                                                                  j*hr_ps:(j+1)*hr_ps]

        return ret_hr, ret_lr, mach

    def readH5_wrapper(filename):
        hr, lr, mach = tf.py_function(readH5, [filename], (tf.float32, tf.float32, tf.float32))
        mach = mach*tf.ones(num_patches**2*lr_sim_size)
        ret = {'x': hr, 'y':lr, 'mach':mach}
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
    dataset = dataset.shuffle(num_patches**2*lr_sim_size*len(sim_list)).repeat()
    dataset = dataset.batch(batch_size*strategy.num_replicas_in_sync)
    dataset = strategy.experimental_distribute_dataset(dataset)
    return dataset

def get_dataset_dmh(dset_dir, redshifts, tilesize, inputs, targets, batch_size, strategy):
    def randomize_tile(xd, xv, y):
        nmir = np.random.randint(0, 2)
        nrot = np.random.randint(0, 4)
        if nmir:
            xd = np.fliplr(xd)
            xv = np.fliplr(xv)
            y = np.fliplrt(y)

        xd = np.rot90(xd, k=nrot)
        xv = np.rot90(xv, k=nrot)
        y = np.rot90(y, k=nrot)
        return (xd, xv, y)
    
    def load_new_slice(dset_dir, redshifts, inputs, targets):
        rz = np.random.choice(redshifts, 1)[0]
        rx = np.random.choice(['x', 'y'], 1)[0]
        ri = np.random.randint(0, 10)  # 10 slices each for FIREbox
        q = np.random.randint(0, 49) # which sub-tile
        xd = np.load('{}snapshot_{}/maps/p{}_{}_{}.d.{}.npy'.format(dset_dir, rz, inputs, rx, ri, q))
        xv = np.load('{}snapshot_{}/maps/p{}_{}_{}.v.{}.npy'.format(dset_dir, rz, inputs, rx, ri, q))
        yd = np.load('{}snapshot_{}/maps/p{}_{}_{}.d.{}.npy'.format(dset_dir, rz, targets, rx, ri, q))
        maximum = xd.shape[0] - int(tilesize)
        
        return xd, xv, yd, maximum

        #self.xd, self.xv, self.yd = self.randomize_tile(self.xd, self.xv, self.yd)


    def _slice(maximum):
        ptx, pty = np.random.randint(low=0, high=maximum, size=(2,))
        return np.s_[ptx: ptx + tilesize, pty: pty + tilesize]
    
    def _read_files(x):
        input_d, input_v, target_d = x.numpy()
        xd = np.load(input_d)
        xv = np.load(input_v)
        yd = np.load(target_d)
        
        num_patches = (xd.shape[0]//tilesize)
        xd = xd.reshape(*(xd.shape), 1)
        xv = xv.reshape(*(xv.shape), 1)
        yd = yd.reshape(*(yd.shape), 1)
        
        xd = tf.image.resize(xd, size=[128*num_patches, 128*num_patches])
        xv = tf.image.resize(xv, size=[128*num_patches, 128*num_patches])
        yd = tf.image.resize(yd, size=[128*num_patches, 128*num_patches])

        ret_x = np.zeros([num_patches**2,128,128,1])
        ret_y = np.zeros([num_patches**2,128,128,2])
        
        x = np.arange(0,xd.shape[0]//128,1)
        ii, jj = np.meshgrid(x,x)
        for i,j in zip(ii.flat, jj.flat):
            idx = i + num_patches * j
            sl = np.s_[i*128: (i+1)*128, j*128: (j+1)*128, 0]
            ret_y[idx, :, :, 0] = yd[sl]
            ret_y[idx, :, :, 1] = xv[sl]
            ret_x[idx, :, :, 0] = xd[sl]
        return ret_x, ret_y
    
    def read_files(x):
        x, y = tf.py_function(_read_files, [x], (tf.float32, tf.float32))
        ret = {'x': x, 'y':y}
        return ret

    input_d = [f'{dset_dir}snapshot_{r}/maps/p{inputs}_*_*.d.*.npy' for r in redshifts]
    input_v = [f'{dset_dir}snapshot_{r}/maps/p{inputs}_*_*.v.*.npy' for r in redshifts]
    target_d = [f'{dset_dir}snapshot_{r}/maps/p{targets}_*_*.d.*.npy' for r in redshifts]
    
    for i in range(len(redshifts)):
        input_d[i] = sorted(glob.glob(input_d[i]))
        input_v[i] = sorted(glob.glob(input_v[i]))
        target_d[i] = sorted(glob.glob(target_d[i]))
    
    target_d = list(itertools.chain(*target_d))
    input_v = list(itertools.chain(*input_v))
    input_d = list(itertools.chain(*input_d))
    
    file_list = [[input_d[i], input_v[i], target_d[i]] for i in range(len(input_d))]
    
    random.shuffle(file_list)
    
    dataset = tf.data.Dataset.from_tensor_slices(file_list)
    dataset = dataset.map(lambda x: read_files(x))
    dataset = dataset.unbatch()
    dataset = dataset.batch(batch_size*strategy.num_replicas_in_sync)
    #dataset = strategy.experimental_distribute_dataset(dataset)
    return dataset
