#!/bin/bash

# n: Calculate the total number of jobs for SS2D model

calculate_total_jobs() {
    J0_values=(1)  # Define J0 values
    J1_values=($(seq 0 0.1 2))  # Define J1 values
    J2_values=($(seq 0 0.1 2))  # Define J2 values

    num_J0=${#J0_values[@]}
    num_J1=${#J1_values[@]}
    num_J2=${#J2_values[@]}

    total_jobs=$((num_J0 * num_J1 * num_J2))  # Calculate total jobs
}

# Function to calculate J0, J1, and J2 from a single integer input
calculate_parameters() {
    calculate_total_jobs
    local task_id=$1

    i=$((task_id / (num_J1 * num_J2)))
    j=$((task_id % (num_J1 * num_J2) / num_J2))
    k=$((task_id % num_J2))

    J0=$(printf "%.3f" "${J0_values[$i]}")
    J1=$(printf "%.3f" "${J1_values[$j]}")
    J2=$(printf "%.3f" "${J2_values[$k]}")
}

# Echo all jobs to confirm the parameters
echo_jobs() {
    for i in $(seq 0 $((total_jobs - 1))); do
        calculate_parameters "$i"
        echo "J0=${J0}, J1=${J1}, J2=${J2}"
    done
    echo "Total number of jobs for SS2D model: $total_jobs"
}

# Core job function for SS2D model
run_job() {
    local task_id=$1
    local project_dir=$2
    local n_cpu=$3

    calculate_parameters "$task_id"

    LT=1
    SWEEPS=1000000
    EPOCH=5000
    M=1
    model_name="SS2D"
    log_dir="${project_dir}/job/log/${model_name}"
    [ ! -d "$log_dir" ] && mkdir -p "$log_dir" && echo "Created log directory $log_dir"
    log_file="${log_dir}/J0_${J0}_J1_${J1}_J2_${J2}_output.log"
    link_dir="${project_dir}/job/link/${model_name}"
    symbolic_link="${link_dir}/J0_${J0}_J1_${J1}_J2_${J2}"

    # Create the link directory if it does not exist
    [ ! -d "$link_dir" ] && mkdir -p "$link_dir" && echo "Created link directory $link_dir"

    echo "Symbolic link for SS2D model: $symbolic_link"
    echo "Project directory: $project_dir"

    cd "$project_dir/python/rmsKit" || return

    if [ "$n_cpu" -gt 0 ]; then
        echo "Using $n_cpu CPUs"
        export OMP_NUM_THREADS=$n_cpu
        export MKL_NUM_THREADS=$n_cpu
    else
        echo "Using all available CPUs"
    fi
    source /opt/materiapps-intel/env.sh
    source ~/worms/myenv/bin/activate

    python -u optimize_loc.py -m $model_name -o Adam -lr 0.003 -e "$EPOCH" -M "$M" -lt "$LT" \
        -J0 "$J0" -J1 "$J1" -J2 "$J2" \
        -n "$n_cpu" --symoblic_link "$symbolic_link" \
        --stdout >> "$log_file"

    echo "Finished SS2D model job with J0=${J0}, J1=${J1}, J2=${J2} in CPU ${n_cpu}"

    python -u -m run_worm -m $model_name --path "$symbolic_link" -s "$SWEEPS" --original -n "$n_cpu" --stdout >> "$log_file"

    echo "Finished running worm with original hamiltonian"

    python -u -m run_worm -m $model_name --path "$symbolic_link" -s "$SWEEPS" -n "$n_cpu" --stdout >> "$log_file"

    echo "Finished running worm with optimized hamiltonian"
}

# Initialization
calculate_total_jobs
echo_jobs
echo "Sourcing script ss2d.sh"
export -f run_job
export -f calculate_parameters
export -f calculate_total_jobs
