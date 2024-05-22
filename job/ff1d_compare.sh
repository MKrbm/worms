#!/bin/bash

# n: Calculate the total number of jobs for FF1D model

calculate_total_jobs() {
    seed_values=($(seq 0 1 999))  # Define seed values

    total_jobs=${#seed_values[@]}  # Calculate total jobs
}

# Function to calculate seed and other parameters from a single integer input
calculate_parameters() {
    calculate_total_jobs
    local task_id=$1

    seed="${seed_values[task_id]}"
}

# Echo all jobs to confirm the parameters
echo_jobs() {
    for i in $(seq 0 $((total_jobs - 1))); do
        calculate_parameters "$i"
        echo "Seed=${seed}"
    done
    echo "Total number of jobs for FF1D model: $total_jobs"
}

# Core job function for FF1D model
run_job() {
    local task_id=$1
    local project_dir=$2
    local n_cpu=$3

    calculate_parameters "$task_id"

    sps=3
    model_name="FF1D"
    log_dir="${project_dir}/job/logs/${model_name}"
    [ ! -d "$log_dir" ] && mkdir -p "$log_dir" && echo "Created log directory $log_dir"
    link_dir="${project_dir}/job/link/${model_name}"
    symbolic_link="${link_dir}/seed_${seed}_sps_${sps}"
    log_file_orth="${log_dir}/seed_${seed}_orth.log"
    log_file_uni="${log_dir}/seed_${seed}_uni.log"


    # Directory change and environment setup (similar to previous scripts)
    cd "$project_dir/python/rmsKit" || return

    # if n_cpu is positive, then use that number of CPUs
    if [ "$n_cpu" -gt 0 ]; then
        echo "Using $n_cpu CPUs"
        export OMP_NUM_THREADS=$n_cpu
        export MKL_NUM_THREADS=$n_cpu
    else
        echo "Using all available CPUs"
    fi
    source /opt/materiapps-intel/env.sh
    source ~/worms/myenv/bin/activate
    M=1000

    # Remove log files if they exist
    [ -f "$log_file_orth" ] && mv "$log_file_orth" "${log_file_orth}.bak" && echo "Renamed existing log file $log_file_orth to ${log_file_orth}.bak"
    [ -f "$log_file_uni" ] && mv "$log_file_uni" "${log_file_uni}.bak" && echo "Renamed existing log file $log_file_uni to ${log_file_uni}.bak"


    # Run jobs with different settings
    python -u optimize_loc.py -m $model_name -o Adam -e 1500 -lr 0.001 -lt 1 -M $M \
        --sps "$sps" --seed "$seed" --stdout --loss mel --dtype float64 \
        --symoblic_link "$symbolic_link" --stdout >> "$log_file_orth"

    python -u optimize_loc.py -m $model_name -o Adam -e 1500 -lr 0.001 -lt 1 -M $M \
        --sps "$sps" --seed "$seed" --stdout --loss mel --dtype complex128 \
        --symoblic_link "$symbolic_link" --stdout >> "$log_file_uni"


    echo "Finished optimization of FF1D model job with seed = ${seed} in CPU ${n_cpu}"

    echo "Cleaning up existing symbolic links $symbolic_link"
}

# Initialization
calculate_total_jobs
echo_jobs
echo "Sourcing script FF1D.sh"
export -f run_job
export -f calculate_parameters
export -f calculate_total_jobs
