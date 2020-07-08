from __future__ import division
import grafic
import numpy as np
import sys
import subprocess
import re
import gc

from my_units import *
from formulas import *

"""
    TURBULENT BOX INITIAL CONDITIONS

    This script generated IC for turbulent box simulation.
    It needs an initial raw velocity field (at the highest initial resolution desired)
    as input and rescales it in the desired way.

    Velocity rescaling options:
        1) Choose a fixed total 1D or 3D Mach number (option 'Mach_number1D' or 'Mach_number3D')
           Input value = 3D Mach number
        2) Assume the cloud obeys Larson's relation (option 'Larson')'
           Input value = not used

    Mass normalisation options:
        1) Choose a fixed total cloud mass (option 'Fixed_mass')
           Input value = mass (in Msun)
        2) Choose a fixed average density (option 'Fixed_density')
           Input value = density (in g/cm3)
        3) Choose a fixed virial parameter value (option 'Virial')
           Input value = virial parameter (1 for a stable cloud)

           In option 3) the virial parameter is determined for a spherical cloud
                        virial parameter = E_turbulence / E_gravity
     		                         = 1 for stable cloud
     	                E_turb = 2 E_kin = M * disp3D^2 = M * (mach3D*cs)^2 = 3 * M mach1D^2 kT/(mu mH)	                     E_grav = (3/5) * G*M^2/R
                        (4pi/15) rho * R^2 G = mach1D cs^2
                        => stable cloud: rho = (15/4pi) * (Mach1D*cs)^2/(a*G*R^2)
     	                => stable cloud: rho = (5/4pi) * (disp3D^2)/(a*G*R^2)

    Use: python rescale_fields.py <levelmax>
                                  <box size [c.u.]>
                                  <temperature>
                                  <mean molecular weight>
                                  <velocity scaling option> <value>
                                  <mass normalisation option> <value>

"""

# MAIN
if __name__=='__main__':

    # ARGUMENTS
    lvl_max = int(sys.argv[1])            # highest refinement level in the ICs
    print('lvl max=', lvl_max)
    size_cu = float(sys.argv[2])          # size of the box (in c.u.)
    print('boxsize', size_cu)
    T = float(sys.argv[3])                # temperature (in K)
    mu = float(sys.argv[4])               # mean molecular weight (in MCs)
    print('T', T, 'mu', mu)
    option_vel = str(sys.argv[5])         # method of rescaling for the velocity
    value_vel = float(sys.argv[6])        # value corresponding to the velocity rescaling
    print(option_vel, value_vel)
    option_mass = str(sys.argv[7])        # method of rescaling for the density
    value_mass = float(sys.argv[8])       # fixed mass (solar mass) or density (cm-3)

    print(option_mass, value_mass)

    size = size_cu*scale_l #cm

    # derived quantities
    res1D = int(2**lvl_max)
    num_cells = (2.0**lvl_max)**3
    T2 = T/mu # K
    c_s = sound_speed(T, mu, units_cgs) #cm/s

    # PROCESS RAW VELOCITY FIELD
    disp_L_cu = 0.0
    disp_L = 0.0
    mach = 0.0

    if option_vel == "Mach_number3D":
        mach = value_vel/np.sqrt(3.)
        disp_L = mach * c_s #cm/s
        disp_L_cu = disp_L / scale_v #cu
    elif option_vel == "Mach_number1D":
        mach = value_vel
        disp_L = mach * c_s #cm/s
        disp_L_cu = disp_L / scale_v #cu
    elif option_vel == "Larson":
        disp_L = disp_larson(size,units_cgs) #cm/s
        mach = disp_L / c_s
        disp_L_cu = disp_L / scale_v #cu
    else:
        message = "ERROR: unknown velocity rescaling option " + option_vel
        sys.exit(message)

    # raw fields are expected to be in the current directory !!!
    raw_files = ['./ic_vel_u','./ic_vel_v','./ic_vel_w']
    new_files = ['ic_u','ic_v','ic_w']
    new_data = []
    for i in range(3):
        raw_ic = grafic.Grafic()
        raw_ic.read(raw_files[i])
        # determine current velocity dispersion (assume average velocity of zero)
        disp_ini = np.sqrt((raw_ic.data**2.).sum() / num_cells)
        # rescale and write new velocity field: v_new = disp_wanted * v_ini/disp_ini
        new_ic=grafic.Grafic()
        new_ic.data = (disp_L_cu/disp_ini) * raw_ic.data # cu
        new_ic.make_header(size)
        new_ic.write(new_files[i])
        new_data.append(new_ic.data)
        gc.collect()

    # check results
    means = []  #cm/s
    stds = []   #cm/s
    sigmas = [] #cm/S
    for i in range(3):
        means.append(np.mean(new_data[i])*scale_v)
        stds.append(np.std(new_data[i])*scale_v)
        sigmas.append(np.sqrt((new_ic.data**2.).sum() / num_cells)*scale_v)

    sigma_3D = np.sqrt(sigmas[0]**2. + sigmas[1]**2. + sigmas[2]**2.) #cm/s

    # CREATE DENSITY AND PRESSURE IC FIELDS

    value_d_cu = 0.0
    value_d = 0.0
    total_mass = 0.0
    alpha_vir = 0.0

    if option_mass == "Fixed_mass":
        total_mass = value_mass*units_cgs['Msun'] #g
        value_d = total_mass / (size**3.0) #g/cm3
        value_d_cu = value_d / scale_d #cu
        alpha_vir = 5.*(disp_L**2)*(size/2.)/(G_cgs*total_mass)
    elif option_mass == "Fixed_density":
        value_d = value_mass #g/cm3
        #value_n = value_mass #H/cc
        #value_d = value_n * units['mH'] * mu # g/cm3
        value_d_cu = value_d / scale_d # cu
        total_mass = value_d * (size**3.0) #g
        alpha_vir = 5.*(disp_L**2)*(size/2.)/(G_cgs*total_mass)
    elif option_mass == "Virial":
        print('WARNING: deriviation of alpha needs to be checked.',\
              'This is not a good estimate for a periodic box!')
        alpha_vir = value_mass #dimemsionless
        value_d = (5.0/(4.0*np.pi)) * (disp_L**2) / (alpha_vir * G_cgs * (size/2.)**2) #g/cm^3
        value_d_cu = value_d_cgs / scale_d #cu
        total_mass = value_d * (size**3) #g
    else:
        message = "ERROR: unknown mass rescaling option " + option_mass
        sys.exit(message)

    # create rho and P file
    ic_d = grafic.Grafic()
    ic_p = grafic.Grafic()
    ic_d.data=np.full((res1D,res1D,res1D),value_d_cu,dtype='f4') #uniform
    ic_p.data=ic_d.data*((c_s/scale_v)**2.) # P = rho * c_s
    ic_d.make_header(size)
    ic_p.make_header(size)
    ic_d.write('ic_d')
    ic_p.write('ic_p')

    # checks
    ic_mass = ic_d.data.sum()*scale_d*((size/res1D)**3) #g
    print ('total mass input', total_mass, 'output', ic_mass)
    print ('density input', value_d, 'output', ic_mass / size**3)
    print ('virial parameter input', alpha_vir, 'output', 5*(size/2.)*sigma_3D**2 / (3*G_cgs*ic_mass))

    # WRITE INFO FILE

    PARAMS=open("../PARAMS.txt", 'w')

    PARAMS.write('scale_l = ' + str(scale_l) + '\n' + \
                 'scale_t = ' + str(scale_t) + '\n' + \
                 'scale_d = ' + str(scale_d) + '\n' + \
                 'scale_v = ' + str(scale_v) + '\n' + \
                 'scale_m = ' + str(scale_m) + '\n')

    PARAMS.write('INPUT params\n' + \
                 'lvl_max = ' + str(lvl_max) + '\n' + \
                 '        = ' + str(res1D) + '^3 cells\n' + \
                 'box size = ' + str(size_cu) + ' c.u.\n' + \
                 '         = ' + str(size/pc_cgs) + ' pc\n' + \
                 '=> resolution = ' + str(size/res1D/pc_cgs) + ' pc\n' + \
                 '              = ' + str(size/res1D/AU_cgs) + ' AU\n' + \
                 'T = ' + str(T) + ' K\n' + \
                 'mu = ' + str(mu) + '\n' + \
                 '=> T_2 = ' + str(T2) + '\n' + \
                 '=> sound speed = ' + str(c_s * 1e-5) + ' km/s\n' + \
                 '               = ' + str(c_s/scale_v) + ' c.u.\n' + \
                 'Velocity scaling method: ' + str(option_vel) +\
                              ' with value ' + str(value_vel) + '\n' + \
                 'Mass normalisation method: ' + str(option_mass) +\
                                ' with value ' + str(value_mass) + '\n')

    PARAMS.write('\nVelocity field properties:\n')

    PARAMS.write('Mean velocity:\n' + \
                 '    mean_u = ' + str(means[0]) + ' cm/s\n' + \
                 '    mean_v = ' + str(means[1]) + ' cm/s\n' + \
                 '    mean_w = ' + str(means[2]) + ' cm/s\n' + \
                 'Std velocity:\n' + \
                 '    std_u = ' + str(stds[0]) + ' cm/s\n' + \
                 '    std_v = ' + str(stds[1]) + ' cm/s\n' + \
                 '    std_w = ' + str(stds[2]) + ' cm/s\n')

    PARAMS.write('expected velocity dispersion = ' + str(disp_L * 1e-5) + ' km/s\n' + \
                 '                             = ' + str(disp_L_cu) + ' c.u.\n' + \
                 'Real ic velocity dispersions:\n' + \
                 '    sigma_u = ' + str(sigmas[0]*1e-5) + ' km/s\n' + \
                 '    sigma_v = ' + str(sigmas[1]*1e-5) + ' km/s\n' + \
                 '    sigma_w = ' + str(sigmas[2]*1e-5) + ' km/s\n' + \
                 '    sigma3D = ' + str(sigma_3D*1e-5) + ' km/s\n' + \
                 '            = ' + str(sigma_3D/scale_v) + ' c.u.\n' + \
                 'Expected Mach number (1D) = ' + str(mach) + '\n' + \
                 '                     (3D) = ' + str(mach*np.sqrt(3.)) + '\n' + \
                 'Real ic Mach number:\n' + \
                 '    mach u = ' + str(sigmas[0]/c_s) + '\n' + \
                 '    mach v = ' + str(sigmas[1]/c_s) + '\n' + \
                 '    mach w = ' + str(sigmas[2]/c_s) + '\n' + \
                 '    mach3D = ' + str(sigma_3D/c_s) + '\n')

    PARAMS.write('\nDensity field properties:\n')

    PARAMS.write('Input density = ' + str(value_d_cu) + ' c.u.\n' + \
                 '              = ' + str(value_d) + ' g/cm^3\n' + \
                 '              = ' + str(n(value_d, mu, units_cgs)) + ' particles/cm^3\n' + \
                 '              = ' + str(value_d * (pc_cgs**3) / Msun_cgs) + ' M_sun/pc^3\n' + \
                 'Real density = ' + str(ic_mass / size**3) + ' g/cm^3\n' + \
                 'Input total mass = ' + str(total_mass/scale_m) + ' c.u.\n' + \
                 '                 = ' + str(total_mass/Msun_cgs) + ' M_sun\n' + \
                 'Real total mass = ' + str(ic_mass/Msun_cgs) + ' Msun\n' + \
                 'Input virial parameter ~ ' + str(alpha_vir) + '\n' + \
                 'Real virial parameter ~ '+ str(5*(size/2.)*sigma_3D**2/(3*G_cgs*ic_mass)) + '\n')

    PARAMS.write('\nOther properties:\n')

    PARAMS.write('t_ff = ' +str(t_ff(value_d, units_cgs)/yr_cgs) + ' yr\n' + \
                 '     = ' +str(t_ff(value_d, units_cgs)/scale_t) + ' c.u.\n' + \
                 't_cross = ' +str(t_cross(size, sigmas[0], units_cgs)/yr_cgs) + ' yr\n' + \
                 '        = ' +str(t_cross(size, sigmas[0], units_cgs)/scale_t) + ' c.u.\n' + \
                 'Jeans length = ' +str(L_Jeans(value_d, T, mu, units_cgs)/pc_cgs) + ' pc\n' + \
                 '             = ' +str(L_Jeans(value_d, T, mu, units_cgs)/AU_cgs) + ' AU\n' + \
                 '             = ' +str(L_Jeans(value_d, T, mu, units_cgs)/scale_l) + ' c.u.\n')

    PARAMS.close()

