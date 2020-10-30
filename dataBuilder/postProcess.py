import sys, os
import numpy as np
import h5py
import yt
from yt.funcs import mylog
mylog.setLevel(40)

from scipy.ndimage import gaussian_filter
import scipy.stats as stats

def downSample(field, downscale=4):
    sigma = downscale /(2*np.pi)
    ret = gaussian_filter(field, sigma=sigma, mode='wrap')[::downscale,::downscale,::downscale]
    return ret

def decompose_field(u, v, w):
    NX, NY, NZ = u.shape
    kx = np.fft.fftfreq(NX)
    ky = np.fft.fftfreq(NY)
    kz = np.fft.rfftfreq(NZ)
    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing="ij")
    k2 = kx3d**2 + ky3d**2 + kz3d**2
    k2[0,0,0] = 1. # to avoid inf. we do not care about the k=0 component

    u_f = np.fft.rfftn(u)
    v_f = np.fft.rfftn(v)
    w_f = np.fft.rfftn(w)

    div_f = (u_f * kx3d +  v_f * ky3d + w_f * kz3d)
    div_overk = div_f / k2
    u_comp = np.fft.irfftn(div_overk * kx3d)
    v_comp = np.fft.irfftn(div_overk * ky3d)
    w_comp = np.fft.irfftn(div_overk * kz3d)

    u_sol = u - u_comp
    v_sol = v - v_comp
    w_sol = w - w_comp

    return np.array([u_sol, v_sol, w_sol]), np.array([u_comp, v_comp, w_comp])

def get_sol_comp_spectra(u,v,w):
    s, c = decompose_field(u, v, w)
    k, Es = spectrum([s[0,:,:,:],s[1,:,:,:],s[2,:,:,:]], dtype='vel')
    k, Ec = spectrum([c[0,:,:,:],c[1,:,:,:],c[2,:,:,:]], dtype='vel')
    return k, Es, Ec


def spectrum(data, dtype='vel'):
    if dtype=='vel':
        u,v,w = data
        nx, ny, nz = u.shape

        Kk = np.zeros((nx,ny,nz))
        for vel in [u, v, w]:
            Kk += 0.5*fft_comp(vel)

    else:
        nx, ny, nz = data.shape

        Kk = fft_comp(data)

    kx = np.fft.fftfreq(nx)*nx
    ky = np.fft.fftfreq(ny)*ny
    kz = np.fft.fftfreq(nz)*nz

    # physical limits to the wavenumbers
    kmin = 1.0
    kmax = nx//2+1

    kbins = np.arange(kmin, kmax, kmin)
    N = len(kbins)

    # bin the Fourier KE into radial kbins
    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing="ij")
    k = np.sqrt(kx3d**2 + ky3d**2 + kz3d**2)

    k = k.flatten()
    Kk = Kk.flatten()
    Abins, _, _ = stats.binned_statistic(k, Kk,
                                     statistic = "sum",
                                     bins = kbins)

    #Abins *= 4. * np.pi / 3. * (kbins[1:]**3 - kbins[:-1]**3)

    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    return kvals, Abins


def fft_comp(u):
    nx, ny, nz = u.shape
    ret = np.fft.fftn(u)
    ret = ret/(nx*ny*nz)

    return np.abs(ret)**2

def getCoveringGrids(snapshot_path, N=256):
    ds_box = yt.load(snapshot_path)
    level=0
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

def saveFields(h5Group, ux, uy, uz, rho):
    rho_mean = np.mean(rho)

    s = np.log(rho/rho_mean)
    s_mean   = np.mean(s)
    s_std2   = np.mean((s-s_mean)**2)
    ux_mean  = np.mean(ux)
    ux_std2  = np.mean((ux-ux_mean)**2)
    uy_mean  = np.mean(uy)
    uy_std2  = np.mean((uy-uy_mean)**2)
    uz_mean  = np.mean(uz)
    uz_std2  = np.mean((uz-uz_mean)**2)

    h5Group.attrs['rho_mean'] = rho_mean
    h5Group.attrs['s_mean']   = s_mean
    h5Group.attrs['s_std2']   = s_std2
    h5Group.attrs['ux_mean']  = ux_mean
    h5Group.attrs['ux_std2']  = ux_std2
    h5Group.attrs['uy_mean']  = uy_mean
    h5Group.attrs['uy_std2']  = uy_std2
    h5Group.attrs['uz_mean']  = uz_mean
    h5Group.attrs['uz_std2']  = uz_std2

    h5Group.create_dataset('ux', data=ux)
    h5Group.create_dataset('uy', data=uy)
    h5Group.create_dataset('uz', data=uz)
    h5Group.create_dataset('s',  data=s)

    k,Es,Ec = get_sol_comp_spectra(ux,uy,uz)
    k,P = spectrum(s, dtype='rho')
    h5Group.create_dataset('Es', data=Es)
    h5Group.create_dataset('Ec', data=Ec)
    h5Group.create_dataset('P', data=P)
    h5Group.create_dataset('k', data=k)

if __name__ == "__main__":
    base_dir = sys.argv[-1]
    print("Processing simulations {}".format(base_dir))
    if base_dir[-1] != os.path.sep:
        base_dir += os.path.sep

    output_dir = base_dir+"processed_data/"
    filename = output_dir+"snapshot.h5"

    if yt.is_root():
    	if not os.path.isdir(output_dir):
        	os.makedirs(output_dir)

    HR_snapshot_path = base_dir + "/HR_run/output_00002/info_00002.txt"
    LR_snapshot_path = base_dir + "/LR_run/output_00002/info_00002.txt"
    print("\tExtracting HR fields")
    ux, uy, uz, rho = getCoveringGrids(HR_snapshot_path, N=256)

    print("\tExtracting LR fields")
    ux_l, uy_l, uz_l, rho_l = getCoveringGrids(LR_snapshot_path, N=64)

    bSave = (np.min(rho) > 0) and (np.min(rho_l))
    if not bSave:
        print("\tNegative density, ignoring simulation")
        print("Done")
        sys.exit()

    print("\tDownsampling fields")
    scale_list = [2,4,8,16]
    print("\t\tVelocity_x")
    ux_filt  = [downSample(ux,  s) for s in scale_list]
    print("\t\tVelocity_y")
    uy_filt  = [downSample(uy,  s) for s in scale_list]
    print("\t\tVelocity_z")
    uz_filt  = [downSample(uz,  s) for s in scale_list]
    print("\t\tDensity")
    rho_filt = [downSample(rho, s) for s in scale_list]

    print("\tSaving fields in {}".format(filename))
    with h5py.File(filename, 'w') as h5File:

        HR = h5File.create_group('/HR')
        saveFields(HR, ux, uy, uz, rho)

        LR = h5File.create_group('/LR')
        saveFields(LR, ux_l, uy_l, uz_l, rho_l)

        for i, s in enumerate(scale_list):
            FILT = h5File.create_group('/FILT{}'.format(s))
            saveFields(FILT, ux_filt[i], uy_filt[i], uz_filt[i], rho_filt[i])

    print("Done")

