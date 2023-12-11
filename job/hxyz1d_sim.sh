#!/bin/bash

# Function to run a job with specific values of J and H

PROJECT_DIR=$(dirname "$(pwd)")

run_job() {
	PROJECT_DIR=$(dirname "$(pwd)")
  # Get the directory of the current script
	lt=1
  local J=$1
  local H=$2
  cd $PROJECT_DIR/python/rmsKit || return
  # source ~/worms/myenv/bin/activate
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  echo "Running job with J=$J and H=$H"
  python -u new_optimize_loc.py \
    -m HXYZ1D -loss mel -o Adam --lattice_type $lt -M 40 -e 2000 -lr 0.005 \
    -Jz 1.0 -Jx $J -Jy $J -hx $H \
    --stdout >> $PROJECT_DIR/job/optimize/J_${J}_H_${H}_output.log

	python -u -m run_worm  -m HXYZ1D -f $PROJECT_DIR/python/rmsKit/array/torch/HXYZ1D_loc/Jx_${J}_Jy_${J}_Jz_1_hx_${H}_hz_0_lt_${lt}/ \
	-s 10000 --original \
	>> $PROJECT_DIR/job/worm/J_${J}_H_${H}_output.log
	# deactivate
}

rm $PROJECT_DIR/job/worm/*
rm $PROJECT_DIR/job/optimize/*
export -f run_job

NUM_CPU=4

# Generate the list of J and H values
generate_values() {
  for J in $(seq -5 0.5 5); do
    for H in $(seq -5 0.5 5); do
      echo $J $H
    done
  done
}

# Run jobs in parallel
generate_values | parallel -j $NUM_CPU --colsep ' ' run_job {1} {2}
