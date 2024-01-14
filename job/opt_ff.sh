#!/bin/bash

#SBATCH --job-name=opt_ff
#SBATCH --output=job_log/ffopt_%A_%a.log
#SBATCH --array=0-500  # Adjust this for the number of jobs and parallelism
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=cpu
n_job_per_run=2  # Adjust this for the number of jobs per run

for i in $(seq 0 $((n_job_per_run - 1))); do
    job_id=$((SLURM_ARRAY_TASK_ID * n_job_per_run + i + 3000)) 
    echo "Running job $job_id"
    python optimize_loc.py -m FF1D -o Adam -e 15000 -lr 0.001 -lt 1 -M 40 \
        --seed "$job_id" --stdout --loss mel
    python optimize_loc.py -m FF1D -o Adam -e 15000 -lr 0.001 -lt 2 -M 40 \
        --seed "$job_id" --stdout --loss mel
    python optimize_loc.py -m FF1D -o Adam -e 15000 -lr 0.001 -lt 3 -M 40 \
        --seed "$job_id" --stdout --loss mel
    python optimize_loc.py -m FF1D -o Adam -e 2000 -lr 0.001 \
        -lt 1 -M 40 --stdout --loss qsmel -L1 6 --seed "$job_id"
done
