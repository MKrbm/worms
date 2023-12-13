#!/bin/bash

# Function to run a job with specific values of J and H
PROJECT_DIR=$(dirname "$(pwd)")

run_job() {
  # Get the directory of the current script
  lt=1
  N_CPU=1
  local PROJECT_DIR=$1
  local Jx=$2
  local Jy=$3
  local H=$4
  cd $PROJECT_DIR/python/rmsKit || return
  # source /opt/materiapps-gcc/env.sh
  # source ~/worms/myenv/bin/activate
  export OMP_NUM_THREADS=$N_CPU
  export MKL_NUM_THREADS=$N_CPU
  echo "Running job with Jx=$Jx, Jy=$Jy and H=$H"
	echo "Hello" 
  python -u new_optimize_loc.py \
    -m HXYZ1D -loss mel -o Adam --lattice_type $lt -M 2 -e 2000 -lr 0.005 \
    -Jz 1.0 -Jx $Jx -Jy $Jy -hx $H \
    -n $N_CPU \
    --symoblic_link $PROJECT_DIR/job/link/Jx_${Jx}_Jy_${Jy}_Jz_1_hx_${H}_hz_0_lt_${lt} \
    --stdout >> $PROJECT_DIR/job/optimize/Jx_${Jx}_Jy_${Jy}_H_${H}_output.log

  python -u -m run_worm  -m HXYZ1D -f $PROJECT_DIR/job/link/Jx_${Jx}_Jy_${Jy}_Jz_1_hx_${H}_hz_0_lt_${lt} \
    -s 1000 --original \
    -n $N_CPU \
    >> $PROJECT_DIR/job/worm/Jx_${Jx}_Jy_${Jy}_H_${H}_output.log

  python -u -m run_worm  -m HXYZ1D -f $PROJECT_DIR/job/link/Jx_${Jx}_Jy_${Jy}_Jz_1_hx_${H}_hz_0_lt_${lt} \
    -s 1000  \
    -n $N_CPU \
    >> $PROJECT_DIR/job/worm/Optimize_Jx_${Jx}_Jy_${Jy}_H_${H}_output.log

  # deactivate
}

rm $PROJECT_DIR/job/worm/*
rm $PROJECT_DIR/job/optimize/*
export -f run_job

NUM_THREAD=5

# Generate the list of Jx, Jy, and H values
generate_values() {
  for i in $(seq -20 4 20); do
    for j in $(seq -20 4 20); do
      for k in 0 5 10; do
        Jx=$(awk -v val=$i 'BEGIN{ printf "%.2f\n", val/10 }')
        Jy=$(awk -v val=$j 'BEGIN{ printf "%.2f\n", val/10 }')
        H=$(awk -v val=$k 'BEGIN{ printf "%.2f\n", val/10 }')
        run_job $PROJECT_DIR $Jx $Jy $H
      done
    done
  done
}

# Run jobs in parallel
generate_values | parallel -j $NUM_THREAD --colsep ' ' run_job {1} {2} {3}
