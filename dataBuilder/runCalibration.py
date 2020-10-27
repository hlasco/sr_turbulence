#!/usr/bin/env python3
import sys, os, glob
import numpy as np

import argparse
from configparser import ConfigParser

def get_tcross(box, temp, mach, mu=2.37):
    kb_cgs = 1.38064852e-16
    mH_cgs = 1.6737236e-24
    myr_cgs = 1e6*3.1556926e7

    cs = np.sqrt(temp * kb_cgs / (mu*mH_cgs)) # cgs

    pc_cgs = 3.0857e18
    t_cross = box * pc_cgs / (np.sqrt(3)*mach * cs)

    return t_cross / myr_cgs

def get_seeds():
    seeds = np.random.randint(low=1000, high=9999, size=3)
    return seeds

def write_jobscript():

    f = open(job_path,'w')
    f.write("#!/bin/bash \n")
    f.write("#SBATCH --partition=teyssier \n")
    f.write("#SBATCH --time=12:00:00 \n")
    f.write("#SBATCH --output=runCalibration.txt \n")
    f.write("#SBATCH --nodes=1 \n")
    f.write("#SBATCH --ntasks-per-node={} \n".format(config['N_TASKS']))
    f.write("#SBATCH --cpus-per-task=1 \n")
    f.write("#SBATCH --hint=nomultithread \n\n")

    tend = get_tcross(float(config['boxsize']),
	              float(config['temp']),
                      float(mach))
    dt = tend

    f.write("{}/prepareSimulation.sh {} {} {} {} {}\n\n".format(
                script_path, base_path, run_dir, config['level_HR'], tend, dt))

    f.write("cd {} \n".format(ic_dir))
    f.write("export OMP_NUM_THREADS={} \n".format(config['N_TASKS']))
    f.write("{}/boxicgen/generate_hydro_ic.sh {} {} {} {} {} {} {} {} > music_logs.txt \n\n".format(
                script_path, mach, config['boxsize'], config['temp'],
                config['level_HR'], config['level_LR'], str(s[0]), str(s[1]), str(s[2])))


    f.write("cd run\n")
    f.write("export OMP_NUM_THREADS=1\n")
    f.write("mpirun simulation namelist.nml > ramses_logs.txt\n\n")

    f.write("python {}/postProcess.py . {}> postProcess_logs.txt\n".format(script_path, nGrid_HR))

    f.write("cd {}\n".format(script_path))
    f.write("./runCalibration.py --config_file {} --sim_id {}\n".format(args.config_file, args.sim_id+1))
    f.close()

if __name__ == "__main__":

    script_path = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description = "Run calibration simulations.")
    parser.add_argument('--config_file', dest='config_file', help="Configuration file.")
    parser.set_defaults(config_file='calibration.ini')
    parser.add_argument('--sim_id', dest='sim_id', type=int, help="ID of the current simulation.")
    parser.set_defaults(sim_id=0)
    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config_file)
    config = config['all']

    base_path = config['BASEPATH']+'/'
    os.makedirs(base_path, exist_ok=True)
    os.chdir(base_path)

    mach_list = config['mach_0'].split(',')
    nGrid_HR = 2**int(config['level_hr'])

    if (args.sim_id>len(mach_list)):
        print('Done')
        sys.exit(0)

    mach = mach_list[args.sim_id]

    s = get_seeds()

    ic_dir = "{}/mach_{}/{}_{}_{}".format(config['SIMTYPE'], mach.zfill(2), str(s[0]), str(s[1]), str(s[2]))
    run_dir = "{}/run".format(ic_dir)

    job_path = "{}/job.sbatch".format(base_path)
    write_jobscript()

    os.system('sbatch job.sbatch')

