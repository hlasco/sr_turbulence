# Dataset Builder

This module can be used to generate a dataset to train the Super-Resolution GAN. It is designed to run on machines using SLURM Workload Manager. 

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

The script runNewSeeds.sh will successively spawn 100 simulations with different random seeds used for their initial condition. It uses a jobscript that is renewed on the completion of a simultion. You may need to update this jobscript to make it compatible with your machine.
