#!/bin/bash

# Source the shared script
source hxyz1d_sim.sh

P=2
PROJECT_DIR=$(dirname "$(pwd)")
NUM_THREAD=45  # Adjust this to the number of parallel jobs you want to run

# Cleanup existing job data
echo "removing $PROJECT_DIR/job/worm/*"
rm $PROJECT_DIR/job/worm/*

echo "PROJECT_DIR: $PROJECT_DIR"
#source /opt/materiapps-intel/env.sh
#source ~/worms/myenv/bin/activate
#export PATH=~/worms/myenv/bin:$PATH
# Use GNU Parallel to run jobs in parallel
seq 0 $(($total_jobs - 1)) | parallel -j $NUM_THREAD run_job {} $PROJECT_DIR $P
