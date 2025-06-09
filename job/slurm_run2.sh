#!/bin/bash

#SBATCH --job-name=ss2d_sim
#SBATCH --output=job_log/logfile_%A_%a.log
#SBATCH --array=0-128   # Adjust the array range as needed
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=cpu

# source blbq1d_sim_dimer.sh
# source blbq1d_sim.sh
# source ss2d_sim.sh
source kh2d_sim.sh
# source ss2d_sim_dimer.sh
# source mg1d_sim.sh

# Set the project directory
PROJECT_DIR=$(dirname "$(pwd)")

# This environment variable is automatically set by Slurm to the current array index
# But we also can compute how many array tasks there are in total
ARRAY_MIN=$SLURM_ARRAY_TASK_MIN
ARRAY_MAX=$SLURM_ARRAY_TASK_MAX
NARRAY=$((ARRAY_MAX - ARRAY_MIN + 1))

# Now that blbq1d_sim.sh is sourced, we have 'calculate_total_jobs' and 'total_jobs' variable
calculate_total_jobs   # This sets 'total_jobs'

# Compute n_job_per_run by dividing total_jobs by the number of tasks in the array
# The + (NARRAY - 1) ensures we round up if total_jobs isn't a multiple of NARRAY
n_job_per_run=$(( (total_jobs + NARRAY - 1) / NARRAY ))
echo "total_jobs = $total_jobs"
echo "Number of array tasks = $NARRAY"
echo "=> Each array task will process up to $n_job_per_run jobs"

# The index of the current array
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Start job distribution
for i in $(seq 0 $((n_job_per_run - 1))); do
    job_id=$((TASK_ID * n_job_per_run + i))

    # If job_id exceeds the total number of jobs, then break
    if [ "$job_id" -ge "$total_jobs" ]; then
        echo "Job $job_id exceeds total number of jobs ($total_jobs). Exiting loop."
        break
    fi

    echo "Running job $job_id (array task = $TASK_ID)"
    # Pass -1 to use all CPU or set a specific number
    run_job "$job_id" "$PROJECT_DIR" 1
done
