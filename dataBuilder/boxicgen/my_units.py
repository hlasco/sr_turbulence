# CONSTANTS
G_cgs = 6.67259e-8 #cm^3 g^-1 s^-2        # gravitational constant
kb_cgs = 1.38064852e-16 # cm2 g s-2 K-1   # Boltzman constant
mH_cgs = 1.6737236e-24 # g                # hydrogen mass
pc_cgs = 3.0857e18 #cm                    # 1 parsec
yr_cgs = 3.1556926e7 #s                   # 1 year
Msun_cgs = 1.989e33 #g                    # solar mass
AU_cgs = 1.49597871e13 #cm                # 1 astronomical unit
Larson_cgs = 1.0e5 #cm/s                  # proportionality constant for Larsons law
units_cgs = {'G':G_cgs, 'kb':kb_cgs, 'mH':mH_cgs, 'pc':pc_cgs, 'AU':AU_cgs, 'yr':yr_cgs, 'Msun':Msun_cgs, 'Larson':Larson_cgs}

# SCALES
# scale_l and scale_t are set fixed
# scale_d is determined by condition G_cu=1
scale_l = pc_cgs
scale_t = yr_cgs * 1.e6 # 1 Myr
scale_m = (1.0/G_cgs) * (scale_l)**3. / ((scale_t)**2)
scale_d = scale_m / (scale_l)**3.
scale_v = scale_l/scale_t
