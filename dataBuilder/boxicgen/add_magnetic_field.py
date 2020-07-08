import sys
import grafic 
import numpy as np

from my_units import *
from formulas import *

# MAIN
if __name__=='__main__':

    # arguments
    lvl_max = int(sys.argv[1])            # refinement level
    size_cu = float(sys.argv[2])             # size of the box (in c.u.)
    mass = float(sys.argv[3])             # mass (in Msun)
    T = float(sys.argv[4])                # temperature (in K)
    mu = float(sys.argv[5])               # mean molecular weight (in MCs)
    Bz = float(sys.argv[6])                # magnetic field strenght in Gauss

    print lvl_max, size_cu, mass, T, mu, Bz

    mass = mass*Msun_cgs
    size = size_cu*scale_l #cm
    density = mass / (size**3.0) #g/cm3
    c_s = sound_speed(T, mu, units_cgs) #cm/s
    res1D = int(2**lvl_max)

    # magnetic field
    #Bz = 1.e-6 # Gauss = sqrt(g/cm s2)
    Bx=0
    By=0
    mu0 = 1. # in cgs (dimensionless)
    v_alf = Bz/np.sqrt(mu0 * density)
    alf_mach = v_alf/c_s
    value_bx_cu = 0.
    value_by_cu = 0.
    value_bz_cu = Bz / np.sqrt(scale_m/(scale_l * scale_t**2)) 
    # Bx left and right
    ic_Blx = grafic.Grafic()
    ic_Blx.data=np.full((res1D,res1D,res1D),value_bx_cu,dtype='f4') #uniform
    ic_Blx.make_header(size)
    ic_Blx.write('ic_bxleft')
    ic_Brx = grafic.Grafic()
    ic_Brx.data=np.full((res1D,res1D,res1D),value_bx_cu,dtype='f4') #uniform
    ic_Brx.make_header(size)
    ic_Brx.write('ic_bxright')
    # By left and right
    ic_Bly = grafic.Grafic()
    ic_Bly.data=np.full((res1D,res1D,res1D),value_by_cu,dtype='f4') #uniform
    ic_Bly.make_header(size)
    ic_Bly.write('ic_byleft')
    ic_Bry = grafic.Grafic()
    ic_Bry.data=np.full((res1D,res1D,res1D),value_by_cu,dtype='f4') #uniform
    ic_Bry.make_header(size)
    ic_Bry.write('ic_byright')
    # Bz left and right
    ic_Blz = grafic.Grafic()
    ic_Blz.data=np.full((res1D,res1D,res1D),value_bz_cu,dtype='f4') #uniform
    ic_Blz.make_header(size)
    ic_Blz.write('ic_bzleft')
    ic_Brz = grafic.Grafic()
    ic_Brz.data=np.full((res1D,res1D,res1D),value_bz_cu,dtype='f4') #uniform
    ic_Brz.make_header(size)
    ic_Brz.write('ic_bzright')

    # WRITE INFO FILE
    PARAMS=open("../PARAMS_mhd.txt", 'w')
    PARAMS.write('INPUT params\n' + \
                 'magnetic field strenght Bx: ' + str(Bx) + ' Gauss\n' +
                 'magnetic field strenght By: ' + str(By) + ' Gauss\n' +
                 'magnetic field strenght Bz: ' + str(Bz) + ' Gauss\n' +
                 'alven speed: ' + str(v_alf) + ' cm/s\n' +
                 'alfvenic mach number: ' + str(alf_mach) + '\n')
    PARAMS.close()

