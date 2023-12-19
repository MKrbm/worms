#!/bin/bash

# n: Calculate the total number of jobs for MG1D model

calculate_total_jobs() {
    J1_values=(1)  # Define J1 values
    J2_values=($(seq 0 0.2 4))  # Define J2 values
    J3_values=($(seq 0 0.2 4))  # Define J3 values

    num_J1=${#J1_values[@]}
    num_J2=${#J2_values[@]}
    num_J3=${#J3_values[@]}

    total_jobs=$((num_J1 * num_J2 * num_J3))  # Calculate total jobs
}

# Function to calculate J1, J2, and J3 from a single integer input
calculate_parameters() {
    calculate_total_jobs
    local task_id=$1

    i=$((task_id / (num_J2 * num_J3)))
    j=$((task_id % (num_J2 * num_J3) / num_J3))
    k=$((task_id % num_J3))

    J1=$(printf "%.3f" "${J1_values[$i]}")
    J2=$(printf "%.3f" "${J2_values[$j]}")
    J3=$(printf "%.3f" "${J3_values[$k]}")
}

# Echo all jobs to confirm the parameters
echo_jobs() {
    for i in $(seq 0 $((total_jobs - 1))); do
        calculate_parameters "$i"
        echo "J1=${J1}, J2=${J2}, J3=${J3}"
    done
    echo "Total number of jobs for MG1D model: $total_jobs"
}

# Core job function for MG1D model
run_job() {
    local task_id=$1
    local project_dir=$2
    local n_cpu=$3

    calculate_parameters "$task_id"

    LT=2
    SWEEPS=1000000
    EPOCH=2000
    M=20
    model_name="MG1D"
    log_file="${project_dir}/job/worm/${model_name}_J1_${J1}_J2_${J2}_J3_${J3}_output.log"
    link_dir="${project_dir}/job/link/${model_name}"
    symbolic_link="${link_dir}/J1_${J1}_J2_${J2}_J3_${J3}"

    # Create the link directory if it does not exist
    [ ! -d "$link_dir" ] && mkdir -p "$link_dir" && echo "Created link directory $link_dir"

    echo "Symbolic link for MG1D model: $symbolic_link"
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


    # Assuming the environment is already activated and required modules loaded
    python -u optimize_loc.py -m $model_name -o Adam -lr 0.007 -e "$EPOCH" -M "$M" -lt "$LT" \
        -J1 "$J1" -J2 "$J2" -J3 "$J3" \
        -n "$n_cpu" --symoblic_link "$symbolic_link" \
        --stdout >> "$log_file"

    echo "Finished MG1D model job with J1=${J1}, J2=${J2}, J3=${J3} in CPU ${n_cpu}"

    python -u -m run_worm -m $model_name --path "$symbolic_link" -s "$SWEEPS" --original -n "$n_cpu" --stdout >> "$log_file"

    python -u -m run_worm -m $model_name --path "$symbolic_link" -s "$SWEEPS" -n "$n_cpu" --stdout >> "$log_file"

    echo "Finished BLBQ model job with J0=${J0}, J1=${J1}, hz=${hz} and hx=${hx} in CPU ${n_cpu}"

    echo "Cleaning up existing symbolic links $symbolic_link"
}

# Initialization
calculate_total_jobs
echo_jobs
echo "Sourcing script mg1d.sh"
export -f run_job
export -f calculate_parameters
export -f calculate_total_jobs
