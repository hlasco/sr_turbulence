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

    t_cross = t_cross / myr_cgs

    return t_cross

def get_seeds():
    seeds = np.random.randint(low=1000, high=9999, size=3)
    return seeds

def write_jobscript_header(f):
    f.write("#!/bin/bash \n")
    for key in config_job.keys():
        if key != 'basepath':
            f.write("#SBATCH --{}={} \n".format(key, config_job[key]))

def write_jobscript():
    f = open(job_path,'w')
    write_jobscript_header(f)
    f.write("#SBATCH --output=runBuildDataset.txt \n\n")

    tend = get_tcross(float(config_sim['boxsize']),
                      float(config_sim['temp']),
                      float(mach))

    f.write("{}/prepareSimulation.sh {} {} {} {:.4f}\n\n".format(
                script_path, base_path, run_dir_HR, config_sim['level_HR'], tend))

    f.write("{}/prepareSimulation.sh {} {} {} {:.4f}\n\n".format(
                script_path, base_path, run_dir_LR, config_sim['level_LR'], tend))

    f.write("cd {} \n".format(ic_dir))
    f.write("export OMP_NUM_THREADS={} \n".format(config_job['ntasks-per-node']))
    f.write("{}/boxicgen/generate_hydro_ic.sh {} {} {} {} {} {} {} {} > music_logs.txt \n\n".format(
                script_path, mach, config_sim['boxsize'], config_sim['temp'],
                config_sim['level_HR'], config_sim['level_LR'], str(s[0]), str(s[1]), str(s[2])))


    f.write("cd HR_run\n")
    f.write("export OMP_NUM_THREADS=1\n")


    run_cmd = 'srun'
    if config_sim['machine']=='hydra':
        run_cmd = 'mpirun'

    f.write("{} simulation namelist.nml > ramses_logs.txt\n\n".format(run_cmd))

    f.write("cd ../LR_run\n")
    f.write("export OMP_NUM_THREADS=1\n")
    f.write("{} simulation namelist.nml > ramses_logs.txt\n\n".format(run_cmd))

    f.write("cd ..\n")
    f.write("python {}/postProcess.py --base_dir {} --level_HR {} --level_LR {} > postProcess_logs.txt\n".format(script_path, '.', config_sim['level_HR'], config_sim['level_LR']))

    f.write("rm -rf HR_run/output_00001\n")
    f.write("rm -rf HR_run/output_00002\n")
    f.write("rm -rf ic_box3\n")
    f.write("rm -rf LR_run/output_00001\n")
    f.write("rm -rf LR_run/output_00002\n")
    f.write("rm -rf ic_box3\n")

    f.write("cd {}\n".format(script_path))
    f.write("./buildDataset.py --config_file {} --sim_id {}\n".format(args.config_file, args.sim_id+1))
    f.close()

if __name__ == "__main__":

    script_path = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description = "Run simulations and post process simulations of compressible turbulence at various mach number.")
    parser.add_argument('--config_file', dest='config_file', help="Configuration file.")
    parser.set_defaults(config_file='config.ini')
    parser.add_argument('--sim_id', dest='sim_id', type=int, help="ID of the current simulation.")
    parser.set_defaults(sim_id=0)
    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config_file)

    machine = config['sim']['machine']

    config_job = config[machine]
    config_sim = config['sim']

    base_path = config_job['basepath']+'/'
    os.makedirs(base_path, exist_ok=True)
    os.chdir(base_path)

    mach_list = config_sim['mach_0'].split(',')

    N_sim_per_mach = int(config_sim['nsim_max'])
    N_sim_tot = int(config_sim['nsim_max'])*len(mach_list)

    if (args.sim_id>=N_sim_tot):
        print('Done')
        sys.exit(0)

    mach = mach_list[args.sim_id//N_sim_per_mach]

    s = get_seeds()

    ic_dir = "{}/mach_{}/{}_{}_{}".format(config_sim['simtype'], mach.zfill(2), str(s[0]), str(s[1]), str(s[2]))
    run_dir_HR = "{}/HR_run".format(ic_dir)
    run_dir_LR = "{}/LR_run".format(ic_dir)

    job_path = "{}/job_buildDataset.sbatch".format(base_path)

    write_jobscript()

    os.system('sbatch job_buildDataset.sbatch')

