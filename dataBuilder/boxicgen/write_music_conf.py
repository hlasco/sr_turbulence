''' Script the generate the necessary MUSIC config files used to generate the raw velocity field.
    The only important parameters are:
    - refinement level (levelmin = levelmax), we only need the raw field for the highest resolution
    - powerspectrum (nspec=-4) and transfer function (transfer=simpleturb, which we defined to be 1)
    - random number seed
    - name and type of the output file
    Other parameters are irrelevant '''

import sys

if __name__=='__main__':

    res = int(sys.argv[1])       # levelmax (resolution = 2^res)
    comp = sys.argv[2]           # string to indicate the component of the velocity field
    seed = sys.argv[3]           # seed for the random perturbations

    # write config file for velocity comp
    file_name = 'music_ic_' + comp + '.conf'
    f_c = open(file_name, 'w')

    f_c.write('[setup]\n')
    f_c.write(\
        'boxlength    = 100\n' + \
        'zstart       = 50\n' + \
        'levelmin     = ' + str(res) + '\n' + \
        'levelmin_TF  = ' + str(res) + '\n' + \
        'levelmax     = ' + str(res) + '\n' + \
        'padding      = 8\n' + \
        'overlap      = 4\n' + \
        'ref_center   = 0.5, 0.5, 0.5\n' + \
        'ref_extent   = 0.2, 0.2, 0.2\n' + \
        'align_top    = no\n' + \
        'baryons      = no\n' + \
        'use_2LPT     = no\n' + \
        'use_LLA      = no\n' + \
        'periodic_TF  = yes\n' + \
        '\n')

    f_c.write('[cosmology]\n')
    f_c.write(\
        'Omega_m      = 0.276\n' + \
        'Omega_L      = 0.724\n' + \
        'w0           = -1.0\n' + \
        'wa           = 0.0\n' + \
        'Omega_b      = 0.045\n' + \
        'H0           = 70.3\n' + \
        'sigma_8      = 0.811\n' + \
        'nspec        = -4\n' + \
        'transfer     = simpleturb\n' + \
        '\n')

    f_c.write('[random]\n')
    f_c.write('seed[' + str(res) + '] = ' + seed + '\n' + '\n')

    f_c.write('[output]\n')
    f_c.write(\
        '##Grafic2 compatible format for use with RAMSES\n' + \
        '#option ramses_nml=yes writes out a startup nml file\n' + \
        'format      = grafic2\n' + \
        'filename    = raw_ic_' + comp + '\n' + \
        'ramses_nml  = no\n' + \
        '\n')
    
    f_c.write('[poisson]\n')
    f_c.write(\
        'fft_fine    = yes\n' + \
        'accuracy    = 1e-5\n' + \
        'pre_smooth  = 3\n' + \
        'post_smooth = 3\n' + \
        'smoother    = gs\n' + \
        'laplace_order = 6\n' + \
        'grad_order  = 6\n' + \
        '\n')

    f_c.close()
