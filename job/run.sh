#!/bin/bash

# Source the shared script
source hxyz1d_sim.sh

N_CPU=1
PROJECT_DIR=$(dirname "$(pwd)")
NUM_THREAD=5  # Adjust this to the number of parallel jobs you want to run

# Cleanup existing job data
echo "removing $PROJECT_DIR/job/worm/*"
rm $PROJECT_DIR/job/worm/*

echo "PROJECT_DIR: $PROJECT_DIR"
# source /opt/materiapps-gcc/env.sh
# source ~/worms/myenv/bin/activate

# Use GNU Parallel to run jobs in parallel
seq 0 $(($total_jobs - 1)) | parallel -j $NUM_THREAD run_job {} $PROJECT_DIR $N_CPU
# deactivate
