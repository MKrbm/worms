#!/bin/bash

# Function to run a job with specific values of J and H

PROJECT_DIR=$(dirname "$(pwd)")

run_job() {
	PROJECT_DIR=$(dirname "$(pwd)")
  # Get the directory of the current script
	lt=1
	N_CPU=4
  local J=$1
  local H=$2
  cd $PROJECT_DIR/python/rmsKit || return
  # source /opt/materiapps-gcc/env.sh
  # source ~/worms/myenv/bin/activate
  export OMP_NUM_THREADS=$N_CPU
  export MKL_NUM_THREADS=$N_CPU
  echo "Running job with J=$J and H=$H"
  python -u new_optimize_loc.py \
    -m HXYZ1D -loss mel -o Adam --lattice_type $lt -M 3 -e 1000 -lr 0.005 \
    -Jz 1.0 -Jx $J -Jy $J -hx $H \
		-n $N_CPU \
		--symoblic_link $PROJECT_DIR/job/link/Jx_${J}_Jy_${J}_Jz_1_hx_${H}_hz_0_lt_${lt} \
    --stdout >> $PROJECT_DIR/job/optimize/J_${J}_H_${H}_output.log

	python -u -m run_worm  -m HXYZ1D -f $PROJECT_DIR/job/link/Jx_${J}_Jy_${J}_Jz_1_hx_${H}_hz_0_lt_${lt} \
		-s 10000 --original \
		-n $N_CPU \
		>> $PROJECT_DIR/job/worm/J_${J}_H_${H}_output.log
	# deactivate
}

rm $PROJECT_DIR/job/worm/*
rm $PROJECT_DIR/job/optimize/*
export -f run_job

NUM_THREAD=2

# Generate the list of J and H values
generate_values() {
  for J in $(seq -5 0.5 5); do
    for H in $(seq -5 0.5 5); do
      echo $J $H
    done
  done
}

# Run jobs in parallel
generate_values | parallel -j $NUM_THREAD --colsep ' ' run_job {1} {2}
