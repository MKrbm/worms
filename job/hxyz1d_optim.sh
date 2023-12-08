#!/bin/bash

# Function to run a job with specific values of J and H
run_job() {
	PROJECT_DIR=/Users/keisukemurota/Documents/todo/worms
  local J=$1
  local H=$2
  cd $PROJECT_DIR/python/rmsKit || return
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  echo "Running job with J=$J and H=$H"
  python -u new_optimize_loc.py \
    -m HXYZ1D -loss mel -o Adam --lattice_type 1  -M 10 -e 100 -lr 0.005 \
    -Jz 1.0 -Jx $J -Jy $J -hx $H \
    --stdout > $PROJECT_DIR/job/optimize/J_${J}_H_${H}_output.log
}

export -f run_job

NUM_CPU=6

# Generate the list of J and H values
generate_values() {
	for H in $(seq -10 0.5 10); do
		for J in $(seq -10 0.5 10); do
      echo $J $H
    done
  done
}

# Run jobs in parallel
generate_values | parallel -j $NUM_CPU --colsep ' ' run_job {1} {2}
