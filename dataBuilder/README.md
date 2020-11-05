# Dataset Builder

This module can be used to generate a dataset to train for the super-resolution problem applied to 3D compressible turbulence. It is designed to run on machines using SLURM Workload Manager. 

## Requirements

### [RAMSES](https://bitbucket.org/rteyssie/ramses/src/master/)

Fortran90 compiler, OpenMPI

### [MUSIC](https://bitbucket.org/ohahn/music/src/master/)

gcc, FFTW, GSL, HDF5 (optionnal)

### [Boxicgen](https://bitbucket.org/TineColman/boxicgen/src/master/)

python3

### Post-processing

Ramses output are processed with the [yt](https://github.com/yt-project/yt) library, converted into uniform gris then filtered to generate high and low resolution snapshots

## Building the code

The script setup.sh will compile Ramses, MUSIC and Boxicgen.

## Running the code

The script buildDataset.py will successively run pairs of HR/LR simulations at various initial Mach numbers, initialized with random seeds. Simulations are evolved during one turbulent crossing-time, which corresponds to a decay of kinetic energy of about a half. We assume that turbulence is then fully developed.
Each simulation will be launched with its own jobscript after the completion of the previous one.

The main parameters of the simulations can be controlled through the config.ini parameter file.
