#!/bin/bash

#SBATCH -A m4392
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=64
#SBATCH --output="/pscratch/sd/r/ritesh11/diag_logs/slurm-%j.out"
#SBATCH --error="/pscratch/sd/r/ritesh11/diag_logs/slurm-%j.out"
#SBATCH --mail-user=ritesh.slurm@gmail.com
#SBATCH --mail-type=ALL


module load pytorch/2.1.0-cu12

nvidia-smi

./finetune.sh 3 SmolLM2-360M-Instruct 5000 8582
