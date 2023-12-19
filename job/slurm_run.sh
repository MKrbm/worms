#!/bin/bash

#SBATCH --job-name=h1d_sim
#SBATCH --output=log/logfile_%A_%a.log
#SBATCH --array=0-128  # Adjust this for the number of jobs and parallelism
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --partition=cpu

# Source the blbl.sh script
# source blbq1d_sim.sh
source hxyz1d_sim.sh

# Set the project directory
PROJECT_DIR=$(dirname "$(pwd)")

# This environment variable is automatically set by Slurm to the current array index
SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID

n_job_per_run=11  # Adjust this for the number of jobs per run

# Run the job (for loop)
for i in $(seq 0 $((n_job_per_run - 1))); do
    job_id=$((SLURM_ARRAY_TASK_ID * n_job_per_run + i)) 
    echo "Running job $job_id"
    run_job "$job_id" "$PROJECT_DIR" -1
done
