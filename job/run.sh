#!/bin/bash

#n: Source the shared script

# source hxyz1d_sim.sh
# source blbq1d_sim.sh
# source mg1d_sim.sh
# source ss2d_sim.sh
# source ff2d_sim.sh
source ff1d_compare.sh

P=1
PROJECT_DIR=$(dirname "$(pwd)")
NUM_THREAD=6  # Adjust this to the number ou parallel jobs you want to run

# Cleanup existing job data
echo removing "$PROJECT_DIR/job/worm/*"
rm "$PROJECT_DIR"/job/worm/*
echo "PROJECT_DIR: $PROJECT_DIR"

# Use GNU Parallel to run jobs in parallel
seq 0 $((total_jobs - 1)) | parallel -j $NUM_THREAD run_job {} "$PROJECT_DIR" $P