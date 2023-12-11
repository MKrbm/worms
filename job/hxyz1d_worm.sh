#!/bin/bash

# Function to run a job with specific values of J and H

run_job() {
  # Get the directory of the current script
  PROJECT_DIR=$(dirname "$(pwd)")
  local J=$1
  local H=$2
  cd $PROJECT_DIR/python/rmsKit || return
  source ~/worms/myenv/bin/activate
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  echo "Running job with J=$J and H=$H"
	python -u -m run_worm  -m HXYZ1D -f $PROJECT_DIR/python/rmsKit/array/torch/HXYZ1D_loc/Jx_${J}_Jy_${J}_Jz_1_hx_${H}_hz_0_lt_1/ -s 10000\
	>> $PROJECT_DIR/job/korm/J_${J}_H_${H}_output.log
	deactivate
}

export -f run_job

NUM_CPU=4

# Generate the list of J and H values
generate_values() {
  for J in $(seq -10 0.5 10); do
    for H in $(seq -10 0.5 10); do
      echo $J $H
    done
  done
}

# Run jobs in parallel
generate_values | parallel -j $NUM_CPU --colsep ' ' run_job {1} {2}
