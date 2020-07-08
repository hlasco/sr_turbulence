import grafic 
import numpy as np
import sys

from my_units import *

"""
    TURBULENT BOX INITIAL CONDITIONS

    This script adds dust files to the ICs of a turbulent box simulation.
    Currently supported:
      * 1 size bin, set dust2gas ratio

    TODO:
      * option to output either dust2gas ratio or dust density
      * multiple grain size bins seperatly
      * multiple size bins with a grain size distribution
      * add to degrade_grafic

    Use: python add_dust.py <refinement level> <box size in c.u.> <hydro index dust> <dust2gas ratio>

"""

if __name__=='__main__':

    # CAREFUL
    # epsilon = d2g / (1+d2g)
    # d2g = epsilon / (1 - epsilon)

    # arguments
    lvl_max = int(sys.argv[1])            # refinement level
    size_cu = float(sys.argv[2])          # size of the box (in c.u.)
    index = int(sys.argv[3])   # number of the dust bin (starting from 1!)
    value_dust2gas = float(sys.argv[4])   # value for the dust-to-gas ratio

    size = size_cu*scale_l #cm
    res1D = int(2**lvl_max)

    #print(value_dust2gas)
    value = value_dust2gas / (1 + value_dust2gas)

    # initialize dust2gas (uniform for now)
    dust_density = np.full((res1D,res1D,res1D), value, dtype='f4')

    # create grafic file
    ic_dust = grafic.Grafic()
    ic_dust.data=dust_density
    ic_dust.make_header(size)
    ic_dust.write('ic_pvar_{:05}'.format(index))

    # write parameter file
    PARAMS=open("../PARAMS_dust{}.txt".format(index), 'w')
    PARAMS.write('INPUT params\n' + \
                 'dust2gas ratio: ' + str(value_dust2gas) + '\n' +
                 'dust fraction: ' + str(value) + '\n')
    PARAMS.close()

