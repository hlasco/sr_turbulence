import sys, os
import numpy as np
import h5py
import yt
from yt.funcs import mylog
mylog.setLevel(40)

from scipy import ndimage

# Gaussian kernel for filtering
filt1D = np.array([0.04997364, 0.13638498, 0.20002636, 0.22723004, 0.20002636, 0.13638498, 0.04997364])
filt1Dx = np.reshape(filt1D, newshape=(-1,1,1))
filt1Dy = np.reshape(filt1D, newshape=(1,-1,1))
filt1Dz = np.reshape(filt1D, newshape=(1,1,-1))
filt3D = filt1Dx * filt1Dy * filt1Dz

def downSample(field):
    ret = ndimage.convolve(field, filt3D, mode='wrap')[::4,::4,::4]
    return ret

def getCoveringGrids(snapshot_path):
    ds_box = yt.load(snapshot_path)
    level=0
    N = 256
    dims = [int(N), int(N), int(N)]
    cube = ds_box.covering_grid(level,
			left_edge=[0.0, 0.0, 0.0],
                        dims=dims,
                        # And any fields to preload (this is optional!)
                        fields=["velocity_x", "velocity_y", "velocity_z", "density"])

    ux = np.array(cube["velocity_x"].in_units('km/s'))
    uy = np.array(cube["velocity_y"].in_units('km/s'))
    uz = np.array(cube["velocity_z"].in_units('km/s'))

    m_p = 1.6726219e-24
    rho = cube["density"].in_units('g / cm**3') / m_p

    return ux, uy, uz, rho

def process_xdmf(res, filename, h5_filename, dims):
    with open(filename, 'w') as xmFile:
        xmFile.write('''<?xml version="1.0" ?>\n''')
        xmFile.write('''<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n''')
        xmFile.write('''<Xdmf Version="2.0">\n''')
        xmFile.write('''\t<Domain>\n''')
        xmFile.write('''\t\t<Grid Name="Box" GridType="Uniform">\n''')
        xmFile.write('''\t\t\t<Topology TopologyType="3DCORECTMesh" NumberOfElements="{} {} {}" />\n'''.format(dims[0], dims[1], dims[2]))
        xmFile.write('''\t\t\t<Geometry GeometryType="ORIGIN_DXDYDZ">\n''')
        xmFile.write('''\t\t\t\t<DataItem Name="origin" Dimensions="3" NumberType="Float" Precision="4" Format="XML">\n''')
        xmFile.write('''\t\t\t\t\t0.0 0.0 0.0\n''')
        xmFile.write('''\t\t\t\t</DataItem>\n ''')
        xmFile.write('''\t\t\t\t<DataItem Name="spacing" Dimensions="3" NumberType="Float" Precision="4" Format="XML">\n''')
        xmFile.write('''\t\t\t\t\t{} {} {}\n'''.format(100./(dims[0]-1), 100./(dims[1]-1), 100./(dims[2]-1)))
        xmFile.write('''\t\t\t\t</DataItem>\n ''')
        xmFile.write('''\t\t\t</Geometry>\n''')

        #HR density
        xmFile.write('''\t\t\t<Attribute AttributeType="Scalar" Center="Node" Name="Density">\n''')
        xmFile.write('''\t\t\t\t<DataItem DataType="Float" Dimensions="{} {} {}" Format="HDF" Precision="8">\n'''.format(dims[0], dims[1], dims[2]))
        xmFile.write('''\t\t\t\t\t{}:/{}/rho/ \n'''.format(h5_filename, res))
        xmFile.write('''\t\t\t\t</DataItem>\n ''')
        xmFile.write('''\t\t\t</Attribute>\n''')

        #HR velocity
        xmFile.write('''\t\t\t<Attribute AttributeType="Vector" Center="Node" Name="Velocity">\n''')
        xmFile.write('''\t\t\t\t<DataItem ItemType="Function" Function="join($0, $1, $2)" Dimensions="{} {} {} 3">\n'''.format(dims[0], dims[1], dims[2]))
        xmFile.write('''\t\t\t\t\t<DataItem DataType="Float" Dimensions="{} {} {}" Format="HDF" Precision="8">\n'''.format(dims[0], dims[1], dims[2]))
        xmFile.write('''\t\t\t\t\t{}:/{}/ux/ \n'''.format(h5_filename, res))
        xmFile.write('''\t\t\t\t\t</DataItem>\n ''')
        xmFile.write('''\t\t\t\t\t<DataItem DataType="Float" Dimensions="{} {} {}" Format="HDF" Precision="8">\n'''.format(dims[0], dims[1], dims[2]))
        xmFile.write('''\t\t\t\t\t{}:/{}/uy/ \n'''.format(h5_filename, res))
        xmFile.write('''\t\t\t\t\t</DataItem>\n ''')
        xmFile.write('''\t\t\t\t\t<DataItem DataType="Float" Dimensions="{} {} {}" Format="HDF" Precision="8">\n'''.format(dims[0], dims[1], dims[2]))
        xmFile.write('''\t\t\t\t\t{}:/{}/uz/ \n'''.format(h5_filename, res))
        xmFile.write('''\t\t\t\t\t</DataItem>\n ''')
        xmFile.write('''\t\t\t\t</DataItem>\n ''')

        xmFile.write('''\t\t\t</Attribute>\n''')
        xmFile.write('''\t\t</Grid>\n''')
        xmFile.write('''\t</Domain>\n''')
        xmFile.write('''</Xdmf>\n''')

if __name__ == "__main__":
    base_dir = sys.argv[-1]
    print("Processing simulation {}".format(base_dir))
    if base_dir[-1] != os.path.sep:
        base_dir += os.path.sep

    output_dir = base_dir+"processed_data/"
    filename = output_dir+"snapshot.h5"
    filename_LR_xdmf = output_dir+"low_resolution.xdmf"
    filename_HR_xdmf = output_dir+"high_resolution.xdmf"

    if yt.is_root():
    	if not os.path.isdir(output_dir):
        	os.makedirs(output_dir)

    snapshot_path = base_dir + "output_00002/info_00002.txt"
    print("\tExtracting fields")
    ux, uy, uz, rho = getCoveringGrids(snapshot_path)
    print("\tDownsampling fields")
    print("\t\tVelocity_x")
    ux_filt = downSample(ux)
    print("\t\tVelocity_y")
    uy_filt = downSample(uy)
    print("\t\tVelocity_z")
    uz_filt = downSample(uz)
    print("\t\tDensity")
    rho_filt = downSample(rho)

    print("\tSaving fields in {}".format(filename))
    with h5py.File(filename, 'w') as h5File:
        h5File.create_group('/HR')
        h5File.create_dataset('/HR/ux', data=ux)
        h5File.create_dataset('/HR/uy', data=uy)
        h5File.create_dataset('/HR/uz', data=uz)
        h5File.create_dataset('/HR/rho', data=rho)

        h5File.create_group('/LR')
        h5File.create_dataset('/LR/ux', data=ux_filt)
        h5File.create_dataset('/LR/uy', data=uy_filt)
        h5File.create_dataset('/LR/uz', data=uz_filt)
        h5File.create_dataset('/LR/rho', data=rho_filt)

    print("\tSaving xdmf files")
    process_xdmf('LR', filename_LR_xdmf, "snapshot.h5", [64,64,64])
    process_xdmf('HR', filename_HR_xdmf, "snapshot.h5", [256,256,256])
    print("Done")

