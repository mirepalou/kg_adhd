#!/bin/bash

# SLURM and module commands to start a GPU job on the Snellius cluster

#SBATCH --job-name=biobert_1
#SBATCH --time=00:10:00
#SBATCH --partition=gpu # testing is loginnode01, gpu-short, gpu-medium
#SBATCH --output=%x_%j.out  # output.txt
#SBATCH --error=%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1 # access to one gpu only

echo "## Starting GPU test on $HOSTNAME"

module purge

echo "## Loading module"
module load slurm
module load Python/3.9.5-GCCcore-10.3.0
# module load 2021 --- for python???
# module load CUDA/11.8.0
# module load GCC/9.3
# module load cuDNN/8.6.0.163-CUDA-11.8.0
# module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
# module load PyTorch-Lightning/1.5.9-foss-2021a-CUDA-11.3.1


# activate conda environment in shared drive 
# conda init bash
# source ~/.bashrc
# conda env list
# conda activate pytorch_Unet3d
source ~/biobert_env/bin/activate


# move to dictionary to acces code 
cd /home/mapalou/biobert
WORK_DIR=$(pwd)
echo "## Current dircectory $WORK_DIR"

echo "## Number of available CUDA devices: $CUDA_VISIBLE_DEVICES"

echo "## Checking status of CUDA device with nvidia-smi"
nvidia-smi

# starting the Multiclass script 

wandb enabled



# add # copy script to local scratch directory and change into it
python biobert_rel_pred.py

echo "Script finished, model saved"
