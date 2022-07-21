#!/bin/bash
#SBATCH --time=48:0:00
#SBATCH --nodes=1 # number of nodes
#SBATCH --gres=gpu:1 # GPUs per node
#SBATCH --mem=8000M # memory per node
#SBATCH --array=0-14
#SBATCH --output=./slurm_output/slurm-%A_%a.out
python ./job.py within $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_JOB_ID
