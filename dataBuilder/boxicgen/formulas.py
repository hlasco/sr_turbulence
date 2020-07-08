import numpy as np

# definition dependent factor for the free fall time
FACTOR_tff=15.0/np.pi

# FORMULAS
# sound speed c_s
def sound_speed(T, mu, units):
    return (units['kb']*T/(mu*units['mH']))**0.5

# Free fall time
def t_ff(rho, units):
    return np.sqrt(FACTOR_tff) * np.sqrt(1.0/(units['G']*rho))

# Jeans length
def L_Jeans(rho, T, mu, units):
    return sound_speed(T, mu, units) * t_ff(rho, units)

# Jeans mass
def M_Jeans(rho, T, mu, units):
    return (4.0*np.pi/3.0)*rho*(L_Jeans(rho, T, mu, units)/2.0)**3

# number density
def n(rho, mu, units):
    return rho/(mu*units['mH'])

# Mach number
def Mach(disp, T, mu, units):
    return disp / sound_speed(T, mu, units)

# Gravitational energy of a sphere with mass M and radius R
def E_grav(M, R, units):
    return (3.0/5.0) * units['G'] * M**2.0 / R

# Turbulent energy (= 2 Ekin)
def E_turb(M, disp3D, units):
    return M * (disp3D**2.0)

# velocity dispersion according to Larson's law (a and b give deviation from the standard form)
def disp_larson(size,units,a=1,b=1):
    return a*units['Larson'] * (size/units['pc'])**(b*0.5)

# turbulence crossing time
def t_cross(size, disp1D, units):
    return size/disp1D
