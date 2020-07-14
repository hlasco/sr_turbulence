import tensorflow as tf
import ops
import h5py
import os, sys

from dataBuilder.postProcess import process_xdmf

data_dir = '/home/cluster/hlasco/scratch/boxicgen/'
run_dir = '/home/cluster/hlasco/scratch/generator/'

ckpt_G = run_dir + 'SR-RRDB-G_4X.h5'

generator = tf.keras.models.load_model(ckpt_G, compile=False)

dl = ops.DataLoader(data_dir)
lr, hr = dl.loadSnapshot(idx=0)



print('Calling Network')
sr = generator(lr).numpy()
print('Done')

output_dir = run_dir+"sr_fields/"

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

filename = output_dir + "snapshots.h5"

filename_LR_xdmf = output_dir+"low_resolution.xdmf"
filename_SR_xdmf = output_dir+"super_resolution.xdmf"
filename_HR_xdmf = output_dir+"high_resolution.xdmf"

with h5py.File(filename, 'w') as h5File:
    h5File.create_group('/HR')
    h5File.create_dataset('/HR/ux', data=hr[0,:,:,:,0])
    h5File.create_dataset('/HR/uy', data=hr[0,:,:,:,1])
    h5File.create_dataset('/HR/uz', data=hr[0,:,:,:,2])
    h5File.create_dataset('/HR/rho', data=hr[0,:,:,:,3])

    h5File.create_group('/LR')
    h5File.create_dataset('/LR/ux', data=lr[0,:,:,:,0])
    h5File.create_dataset('/LR/uy', data=lr[0,:,:,:,1])
    h5File.create_dataset('/LR/uz', data=lr[0,:,:,:,2])
    h5File.create_dataset('/LR/rho', data=lr[0,:,:,:,3])

    h5File.create_group('/SR')
    h5File.create_dataset('/SR/ux', data=sr[0,:,:,:,0])
    h5File.create_dataset('/SR/uy', data=sr[0,:,:,:,1])
    h5File.create_dataset('/SR/uz', data=sr[0,:,:,:,2])
    h5File.create_dataset('/SR/rho', data=sr[0,:,:,:,3])

process_xdmf('LR', filename_LR_xdmf, "snapshots.h5", [64,64,64])
process_xdmf('SR', filename_SR_xdmf, "snapshots.h5", [256,256,256])
process_xdmf('HR', filename_HR_xdmf, "snapshots.h5", [256,256,256])
