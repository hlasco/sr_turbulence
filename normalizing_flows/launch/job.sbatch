#! /bin/bash
#SBATCH --job-name=srglow3d
#SBATCH --qos=medium
#SBATCH --partition=volta
##SBATCH --gres gpu:Tesla-V100:1
#SBATCH --gres gpu:1
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32G
##SBATCH --hint=nomultithread
#SBATCH --output logs/srflow_x4_vel_1.txt

module load volta cuda/10.0 cudnn

eval "$(conda shell.bash hook)"
conda activate pyenv
export OMP_NUM_THREADS=1

python ../train_sr.py --config_file settings/config_sr4.ini
