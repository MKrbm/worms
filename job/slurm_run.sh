#!/bin/bash

#SBATCH --job-name=blbq_sim
#SBATCH --output=/log/logfile_%A_%a.log
#SBATCH --array=0-29  # Adjust this for the number of jobs and parallelism
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2  # Set the number of CPUs per task
#SBATCH --mem=4G  # Adjust memory as needed
#SBATCH --time=01:00:00  # Adjust the time limit as needed

# Source the blbl.sh script
# source blbq1d_sim.sh
source hxyz1d_sim.sh

# Set the project directory
PROJECT_DIR=$(dirname "$(pwd)")

# This environment variable is automatically set by Slurm to the current array index
SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID

# Run the job
run_job "$SLURM_ARRAY_TASK_ID" "$PROJECT_DIR" -1
