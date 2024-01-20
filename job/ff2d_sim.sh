#!/bin/bash

# n: Calculate the total number of jobs for FF2D model

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
    echo "Total number of jobs for FF2D model: $total_jobs"
}

# Core job function for FF2D model
run_job() {
    local task_id=$1
    local project_dir=$2
    local n_cpu=$3

    calculate_parameters "$task_id"

    model_name="FF2D"
    log_dir="${project_dir}/job/log/${model_name}"
    [ ! -d "$log_dir" ] && mkdir -p "$log_dir" && echo "Created log directory $log_dir"
    log_file="${log_dir}/seed_${seed}.log"
    link_dir="${project_dir}/job/link/${model_name}"
    symbolic_link="${link_dir}/seed_${seed}"

    # Create the link directory if it does not exist
    [ ! -d "$link_dir" ] && mkdir -p "$link_dir" && echo "Created link directory $link_dir"

    echo "Symbolic link for FF2D model: $symbolic_link"
    echo "Project directory: $project_dir"

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
    SWEEPS=1000000
    M=2

    # Run jobs with different settings
    python -u optimize_loc.py -m $model_name -o Adam -e 15000 -lr 0.001 -lt 1 -M $M \
        --sps 8 --seed "$seed" --stdout --loss mel \
        --symoblic_link "$symbolic_link" --stdout >> "$log_file"


    python -u -m run_worm -m $model_name --path "$symbolic_link" -s $SWEEPS \
        --original -n "$n_cpu" --stdout  >> "$log_file"

    python -u -m run_worm -m $model_name --path "$symbolic_link" -s $SWEEPS \
        -n "$n_cpu" --stdout  >> "$log_file"

    echo "Finished BLBQ model job with seed = ${seed} in CPU ${n_cpu}"

    echo "Cleaning up existing symbolic links $symbolic_link"
}

# Initialization
calculate_total_jobs
echo_jobs
echo "Sourcing script ff2d.sh"
export -f run_job
export -f calculate_parameters
export -f calculate_total_jobs
