#!/bin/bash
# Note: job name is set via sbatch --job-name parameter, not here

# Get the SLURM job ID
SLURM_JOB_ID=${SLURM_JOB_ID:-"unknown"}

# Change to project directory
cd /scratch/general/vast/j.yan/myproj1/chpc_manager

# Run the task executor
python3 /scratch/general/vast/j.yan/myproj1/chpc_manager/task_executor.py --config /scratch/general/vast/j.yan/myproj1/chpc_manager/config.yaml --slurm-job-id $SLURM_JOB_ID
