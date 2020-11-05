#! /bin/bash
#SBATCH --job-name=srglow3d
#SBATCH --qos=medium
#SBATCH --partition=volta
#SBATCH --gres gpu:1
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem=16G
##SBATCH --hint=nomultithread
#SBATCH --output logs/flow_3_16_8_128_cond_32_12_4_vel.txt

module load volta cuda/10.1 cudnn

eval "$(conda shell.bash hook)"
conda activate pyenv
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK 

python train_sr.py --config_file config_sr.ini
