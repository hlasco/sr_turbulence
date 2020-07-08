# Script to calculate density scales and parameters for box simulations

import numpy as np
import matplotlib.pyplot as plt
import sys
from my_units import *
from formulas import *

# number of Jeans lengths (X) and masses (Y) we want to resolve
X=4
Y=int((4*np.pi/3.0) * (X/2.0)**3)
#Y=X**3

# determine the density where LJeans = 4*dx at highest resolution
def critical_density(units, boxlen, lvlmax, T, mu):
    dxmin = boxlen*units['pc']/2.0**lvlmax
    return FACTOR_tff * sound_speed(T, mu, units)**2./(units['G']*(X*dxmin)**2.)

# ---- REFINEMENT STRATEGIES ----

''' Jeans refine: resolve the local Jeans length by X cells
    RETURNS:
        the lvls corresponding to each density in the rho array
        the corresponding cell size
        the corresponding cell mass '''
def jeans_refine(rho, T, mu, units, lvlmin, lvlmax):
    LJ = L_Jeans(rho, T, mu,  units)
    LVL = []
    DX=[]
    for lj in LJ:
        curr_lvl = lvlmin
        dx = boxlen/2**curr_lvl * units['pc']
        while (curr_lvl < lvlmax) and (dx >= lj/X):
            curr_lvl = curr_lvl+1
            dx = boxlen/2**curr_lvl * units['pc']
        DX.append(dx)
        LVL.append(curr_lvl)
    DX = np.array(DX)
    DM = rho*(DX**3)
    return LVL, DX/units_cgs['AU'], DM/ units_cgs['Msun']

''' Variable mass refine: refined if Mcell >= m_ref[lvl]*mass_sph (hacked Jeans refine)
    RETURNS:
        the lvls corresponding to each density in the rho array
        the corresponding cell size
        the corresponding cell mass '''
def variable_mass_refine(rho, units, mass_sph_value, m_refine_array, lvlmin, lvlmax):
    LVL = []
    DX=[]
    DM=[]
    for d in rho:
        curr_lvl = lvlmin
        dx = boxlen/2**curr_lvl * units['pc']
        dm = d*(dx)**3
        while (curr_lvl < lvlmax) and (dm >= mass_sph_value*m_refine_array[curr_lvl-lvlmin]):
            curr_lvl = curr_lvl+1
            dx = boxlen/2**curr_lvl * units['pc']
            dm = d*(dx)**3
        LVL.append(curr_lvl)
        DX.append(dx)
        DM.append(dm)
    DX = np.array(DX)
    DM = np.array(DM)
    return LVL, DX/units_cgs['AU'], DM/ units_cgs['Msun']

''' Fixed mass refine: refined if Mcell >= mass_sph (assume m_ref[i] = m_ref[i-1] = 1)
    RETURNS:
        the lvls corresponding to each density in the rho array
        the corresponding cell size
        the corresponding cell mass '''
def mass_refine(rho, units, mass_sph_value, lvlmin, lvlmax):
    LVL = []
    DX=[]
    DM=[]
    for d in rho:
        curr_lvl = lvlmin
        dx = boxlen/2**curr_lvl * units['pc']
        dm = d*(dx)**3
        while (curr_lvl < lvlmax) and (dm >= mass_sph_value):
            curr_lvl = curr_lvl+1
            dx = boxlen/2**curr_lvl * units['pc']
            dm = d*(dx)**3
        LVL.append(curr_lvl)
        DX.append(dx)
        DM.append(dm)
    DX = np.array(DX)
    DM = np.array(DM)
    return LVL, DX/units_cgs['AU'], DM/ units_cgs['Msun']

#---------------------------------------------------------------------------------------------------
if __name__=='__main__':

    # ---- INPUT CLOUD PARAMETERS ----

    if len(sys.argv) > 1:
        lvlmin = int(sys.argv[1])
        lvlmax = int(sys.argv[2])
        boxlen = float(sys.argv[3])
        boxmass = float(sys.argv[4])
        T = float(sys.argv[5])
        mu = float(sys.argv[6])
    else: #default
        lvlmin = 8          # minimum refinement lvl
        lvlmax = 14         # maximum refinement lvl
        boxlen = 0.25 #pc   # size of the box
        boxmass= 260  #Msun # total mass in the box
        T = 10.0 #K         # temperature
        mu = 2.37           # mean molecular weight (in MCs)

    # DETERMINE CHARACTERISTICS

    print 'X', X, 'Y', Y
    print "resolution: ", boxlen/(2**lvlmax) * pc_cgs/AU_cgs, "AU"
    rhoBox = boxmass*units_cgs['Msun']/(boxlen*units_cgs['pc'])**3
    print 'average box density =', rhoBox, 'g/cc'

    rhoCrit = critical_density(units_cgs, boxlen, lvlmax, T, mu)
    rhoSink = rhoCrit
    rhoClump = rhoSink/10.0
    print 'Critical density =', rhoCrit, 'g/cc'

    LJ_lvlmax = L_Jeans(rhoCrit, T, mu, units_cgs)
    MJ_lvlmax = M_Jeans(rhoCrit, T, mu, units_cgs)
    tff = t_ff(rhoCrit, units_cgs)
    mass_sph = MJ_lvlmax/Y
    seedmass = MJ_lvlmax/units_cgs['Msun']
    print 'Ljeans at critical density:', LJ_lvlmax/units_cgs['AU'], 'AU'
    print 'Mjeans at critical density:', MJ_lvlmax/units_cgs['Msun'], 'Msun'
    print 't_ff at critical density:', tff/units_cgs['yr'], 'yr'

    # determine m_refine array for variable mass refine
    m_refine = np.ones(lvlmax-lvlmin+1)
    lvl = lvlmax-lvlmin-1
    while lvl >= 0:
        m_refine[lvl] = m_refine[lvl+1]*2.0
        lvl = lvl - 1

    # ---- WRITE USEFULL NAMELIST BLOCKS ----

    print '\nNAMELIST params:'

    print '\n&AMR_PARAMS'
    print 'levelmin=', lvlmin
    print 'levelmax=', lvlmax
    print 'ngridtot=150000000'
    print 'nparttot=12000000'
    print 'nexpand=4,4,4,6,6,6,6'
    print 'boxlen=', boxlen

    print '\n&REFINE_PARAMS'
    print 'interpol_var=1'
    print 'interpol_type=0'
    print 'mass_sph=', mass_sph/scale_m, '!c.u., (', mass_sph/units_cgs['Msun'], 'Msun)'
    print 'm_refine=', m_refine, '!(for variable mass refine)'
    print 'jeans_refine=4,4,4 !to force refinement on ICs'
    print 'sink_refine=.true.' 
    print '/'

    print '\n&CLUMPFIND_PARAMS'
    print 'ivar_clump=1'
    print 'rho_clfind=', rhoClump, '!g/cm3'
    print 'clinfo=.true.'
    print '/'

    print '\n&SINK_PARAMS'
    print 'create_sinks=.true.'
    print 'accretion_scheme=\'bondi\''
    print 'clump_core=.true.'
    print 'rho_sink=', rhoSink, '!g/cm3'
    print 'mass_sink_seed=', seedmass, '!Msun'
    print 'merging_timescale=1500 !yr (fixed to 1LC timescale)'
    print '/'

    #---- CALC STUFF FOR MAKING PLOTS ----

    rho = np.logspace(-19,-11,150)
    # jeans lenght and jeans mass
    Ljeans = L_Jeans(rho, T, mu, units_cgs) / units_cgs['AU'] # AU
    Lj4 = Ljeans/X
    Mjeans = M_Jeans(rho, T, mu, units_cgs) / units_cgs['Msun'] #Msun
    Mj64 = Mjeans/Y
    # Determine the resolutions for the cases: Jeans refine, Mass refine, Variable mass refine
    lvl_jeanref, dx_jeanref, dm_jeanref = jeans_refine(rho, T, mu, units_cgs, lvlmin, lvlmax)
    lvl_massref, dx_massref, dm_massref = mass_refine(rho, units_cgs, mass_sph, lvlmin, lvlmax)
    lvl_varmref, dx_varmref, dm_varmref = variable_mass_refine(rho, units_cgs, mass_sph, m_refine,
                                                               lvlmin, lvlmax)

    # ---- MAKE PLOTS ----

    plotsize=(6,5) #(10,9)

    # length resolution plot
    plt.figure(1, figsize=plotsize)
    plt.plot(rho, Ljeans, linewidth=2, label='Jeans length', color='black')
    plt.plot(rho, Lj4, linewidth=2, label='required resolution (L_jeans/'+str(X)+')', color='grey')
    #plt.plot(rho, dx_jeanref, marker='3', label='Jeans refine', color='green')
    #plt.plot(rho, dx_massref, marker='4', label='Fixed mass refine', color='blue')
    plt.plot(rho, dx_varmref, marker=',', label='Variable mass refine', color='red')
    plt.plot((rhoClump, rhoClump), (min(Lj4), max(Ljeans)), label='clump finder threshold',
             linewidth=2, color='orange')
    plt.plot((rhoSink, rhoSink), (min(Lj4), max(Ljeans)), label='sink formation threshold',
             linewidth=2,  color='brown')
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.xlabel("rho (g/cm3)")
    plt.ylabel("length scale (AU)")
    plt.legend(loc ='upper right')
    plt.savefig('length_scales.png')

    # mass resolution plot
    plt.figure(2, figsize=plotsize)
    plt.plot(rho, Mjeans, linewidth=2, label='Jeans mass', color='black')
    plt.plot(rho, Mj64, linewidth=2, label='required resolution (M_jeans/'+str(Y)+')', color='grey')
    #plt.plot(rho, dm_jeanref, marker='3', label='Jeans refine', color='green')
    #plt.plot(rho, dm_massref, marker='4', label='Fixed mass refine', color='blue')
    plt.plot(rho, dm_varmref, marker=',', label='Variable mass refine', color='red')
    plt.plot((rhoClump, rhoClump), (min(Mj64), max(Mjeans)), label='clump finder threshold',
             linewidth=2, color='orange')
    plt.plot((rhoSink, rhoSink), (min(Mj64), max(Mjeans)), label='sink formation threshold',
             linewidth=2,  color='brown')
    plt.plot((min(rho), max(rho)), (MJ_lvlmax/Y/units_cgs['Msun'], MJ_lvlmax/Y/units_cgs['Msun']),
             linewidth=2, label="minimal MJeans/"+str(Y), color='magenta')
    plt.plot((min(rho), max(rho)), (seedmass, seedmass), label="seedmass",
             linewidth=2, color='turquoise')
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.xlabel("rho (g/cm3)")
    plt.ylabel("Mass scale (Msun)")
    plt.legend(loc ='upper right')
    plt.savefig('mass_scales.png')

    # refinement lvl plot
    plt.figure(3, figsize=plotsize)
    plt.plot((rhoBox, rhoBox), (lvlmin-0.5, lvlmax+0.5), label='initial box density',
             linewidth=2, color='grey')
    #plt.plot(rho, lvl_jeanref, marker='3', label='Jeans refine', color='green')
    #plt.plot(rho, lvl_massref, marker='4', label='Fixed mass refine', color='blue')
    plt.plot(rho, lvl_varmref, marker=',', label='Variable mass refine', color='red')
    plt.plot((rhoClump, rhoClump), (lvlmin-0.5, lvlmax+0.5), label='clump finder threshold',
             linewidth=2, color='orange')
    plt.plot((rhoSink, rhoSink), (lvlmin-0.5, lvlmax+0.5), label='sink formation threshold',
             linewidth=2,  color='brown')
    plt.xscale("log")
    plt.grid()
    plt.xlabel("rho (g/cm3)")
    plt.ylabel("Refinement level")
    plt.legend(loc ='upper left')
    plt.savefig('refinement_lvl.png')
