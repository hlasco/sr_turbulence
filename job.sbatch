#! /bin/bash
#SBATCH --job-name=PISRT
#SBATCH --partition=volta
#SBATCH --gres gpu:Tesla-V100:1
#SBATCH --time=23:45:00
#SBATCH --ntasks=1
#SBATCH --mem=30G
##SBATCH --hint=nomultithread
#SBATCH --output logs/train4.txt

module load volta cuda/10.1 cudnn

eval "$(conda shell.bash hook)"
conda activate pyenv
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK 

python train_GAN.py
