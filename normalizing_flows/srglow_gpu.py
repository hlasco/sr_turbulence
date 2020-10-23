import sys, os, glob, h5py
sys.path += ['.']

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from flows.cglow import CGlowFlowSR
from flows.invert import Invert
from flows.cglow.cond_affine_coupling import cond_coupling_nn_glow
from flows.glow.affine_coupling import coupling_nn_glow
from flows.cglow.affine_injector import injector_nn_glow
from flows.cglow.cond_gaussianize import cond_gaussianize
from models.flvm import FlowLVM

nGPUs = len(tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", nGPUs, flush=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

dset_dir = '/home/cluster/hlasco/bulk1/GAN_dataset/'
fList_t  = glob.glob(dset_dir+'/train/*/processed_data/snapshot.h5')
fList_v  = glob.glob(dset_dir+'/validation/*/processed_data/snapshot.h5')
nFiles_t = len(fList_t)

ps_h = 64
ps_l = ps_h//4

# read .h5 file
def readH5(f, n_channels=4):
    f = f.numpy()
    idx = np.random.randint(0,4)
    idy = np.random.randint(0,4)
    idz = np.random.randint(0,4)
    with h5py.File(f, 'r') as fi:
        u_hr = np.array(fi['HR/ux'][idx*ps_h:(idx+1)*ps_h,idy*ps_h:(idy+1)*ps_h,idz*ps_h:(idz+1)*ps_h], dtype=np.float32).reshape(ps_h,ps_h,ps_h,1)
        u_lr = np.array(fi['LR/ux'][idx*ps_l:(idx+1)*ps_l,idy*ps_l:(idy+1)*ps_l,idz*ps_l:(idz+1)*ps_l], dtype=np.float32).reshape(ps_l,ps_l,ps_l,1)
        v_hr = np.array(fi['HR/uy'][idx*ps_h:(idx+1)*ps_h,idy*ps_h:(idy+1)*ps_h,idz*ps_h:(idz+1)*ps_h], dtype=np.float32).reshape(ps_h,ps_h,ps_h,1)
        v_lr = np.array(fi['LR/uy'][idx*ps_l:(idx+1)*ps_l,idy*ps_l:(idy+1)*ps_l,idz*ps_l:(idz+1)*ps_l], dtype=np.float32).reshape(ps_l,ps_l,ps_l,1)
        w_hr = np.array(fi['HR/uz'][idx*ps_h:(idx+1)*ps_h,idy*ps_h:(idy+1)*ps_h,idz*ps_h:(idz+1)*ps_h], dtype=np.float32).reshape(ps_h,ps_h,ps_h,1)
        w_lr = np.array(fi['LR/uz'][idx*ps_l:(idx+1)*ps_l,idy*ps_l:(idy+1)*ps_l,idz*ps_l:(idz+1)*ps_l], dtype=np.float32).reshape(ps_l,ps_l,ps_l,1)
        r_hr = np.array(fi['HR/rho'][idx*ps_h:(idx+1)*ps_h,idy*ps_h:(idy+1)*ps_h,idz*ps_h:(idz+1)*ps_h], dtype=np.float32).reshape(ps_h,ps_h,ps_h,1)
        r_lr = np.array(fi['LR/rho'][idx*ps_l:(idx+1)*ps_l,idy*ps_l:(idy+1)*ps_l,idz*ps_l:(idz+1)*ps_l], dtype=np.float32).reshape(ps_l,ps_l,ps_l,1)

        m_vel = np.mean([u_lr, v_lr, w_lr])
        s_vel = np.std([u_lr, v_lr, w_lr])

        r_hr = np.log10(r_hr)
        r_lr = np.log10(r_lr)

        m_r = np.mean(r_lr)
        s_r = np.std(r_lr)

        u_hr = (u_hr - np.mean(u_lr))/np.std(u_lr)
        u_lr = (u_lr - np.mean(u_lr))/np.std(u_lr)

        v_hr = (v_hr - np.mean(v_lr))/np.std(v_lr)
        v_lr = (v_lr - np.mean(v_lr))/np.std(v_lr)

        w_hr = (w_hr - np.mean(w_lr))/np.std(w_lr)
        w_lr = (w_lr - np.mean(w_lr))/np.std(w_lr)

        r_hr = (r_hr - m_r)/s_r
        r_lr = (r_lr - m_r)/s_r

        ret_lr = np.concatenate((u_lr, v_lr, w_lr, r_lr), axis=-1)
        if n_channels==4:
            ret_hr = np.concatenate((u_hr, v_hr, w_hr, r_hr), axis=-1)
        elif n_channels==3:
            ret_hr = np.concatenate((u_hr, v_hr, w_hr), axis=-1)
        elif n_channels==1:
            ret_hr = r_hr
        return ret_lr, ret_hr



def readH5_wrapper(filename, n_channels=4):
    # Assuming your data and labels are float32
    # Your input is parse_function, who arg is filename, and you get X and y as output
    # whose datatypes are indicated by the tuple argument
    lr, hr = tf.py_function(readH5, [filename, n_channels], (tf.float32, tf.float32))
    return hr, lr

batch_size = 1
n_channels = 3
# Create dataset of filenames.
X_train_ds = tf.data.Dataset.from_tensor_slices(fList_t)
X_train_ds = X_train_ds.map(lambda x: readH5_wrapper(x, n_channels=n_channels))
X_train_ds = X_train_ds.batch(batch_size).repeat()


kwargs_nn={'dim':3, 'min_filters':8, 'max_filters':128, 'num_blocks':0}

affine_coupling = coupling_nn_glow(**kwargs_nn)
cond_coupling = cond_coupling_nn_glow(**kwargs_nn)
injector = injector_nn_glow(**kwargs_nn)
parametrizer = cond_gaussianize(**kwargs_nn)
inpt_shape = tf.TensorShape((None,None,None,None,n_channels))
cond_shape = tf.TensorShape((None,None,None,None,4))
glow = Invert(CGlowFlowSR(
                   upfactor=2,
                   num_layers=3,
                   depth=16,
                   cond_shape=cond_shape,
                   cond_filters=32,
                   cond_resblocks=12,
                   cond_blocks=4,
                   cond_coupling_nn_ctor=cond_coupling,
                   injector_nn_ctor=injector,
                   parameterize_ctor=parametrizer))

learning_rate = 2.0e-4
prior = tfp.distributions.Normal(loc=0.0, scale=1.0)
opt_flow = tf.keras.optimizers.Adam(learning_rate=learning_rate)
opt_cond = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model = FlowLVM(glow, prior,
                num_bins=16,
                input_shape=inpt_shape,
                optimizer_flow=opt_flow,
                optimizer_cond=opt_cond)
print(model.param_count())

model._init_checkpoint()
run_dir = '/home/cluster/hlasco/scratch/srflow_3d/flow_3_16_8_128_cond_32_12_4_VEL/'
model.create_checkpoint_manager(run_dir)
#model.load('/home/cluster/hlasco/scratch/srflow_3d/3_16_8_128_1/model',save_num=18)

for i in range(41):
    print('\nEpochs {}-{}:'.format(10*i, 10*(i+1)), flush=True)
    bInit = (i==0)
    model.train(X_train_ds, steps_per_epoch=1000, num_epochs=10, conditional=True, init=bInit)
    print('\nSaving model',  flush=True)
    model.save(run_dir + 'model')

