#!/bin/bash

# n: Calculate the total number of jobs for MG1D model
calculate_total_jobs() {
    J1_values=(1)                         # Define J1 values
    J2_values=($(seq 0 0.5 8.0))          # Define J2 values
    J3_values=($(seq 0 0.5 8.0))          # Define J3 values

    num_J1=${#J1_values[@]}
    num_J2=${#J2_values[@]}
    num_J3=${#J3_values[@]}

    total_jobs=$((num_J1 * num_J2 * num_J3))
}

# Function to compute J1, J2, J3 from a single integer task_id
calculate_parameters() {
    # arrays and total_jobs are assumed set
    local task_id=$1

    i=$(( task_id / (num_J2 * num_J3) ))
    j=$(( task_id % (num_J2 * num_J3) / num_J3 ))
    k=$(( task_id % num_J3 ))

    J1=$(printf "%.3f" "${J1_values[$i]}")
    J2=$(printf "%.3f" "${J2_values[$j]}")
    J3=$(printf "%.3f" "${J3_values[$k]}")
}

# Echo all jobs to verify parameters
echo_jobs() {
    for idx in $(seq 0 $((total_jobs - 1))); do
        calculate_parameters "$idx"
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

    # --- optimization settings (matched to BLBQ style) ---
    LT=-2
    SWEEPS=100000
    EPOCH=5000
    M=70
    model_name="MG1D"

    # Logging setup
    log_dir="${project_dir}/job/log/${model_name}"
    [ ! -d "$log_dir" ] && mkdir -p "$log_dir" && echo "Created log directory $log_dir"
    log_file="${log_dir}/J1_${J1}_J2_${J2}_J3_${J3}_lt_${LT}.log"

    # Symlink directory
    link_dir="${project_dir}/job/link/${model_name}"
    symbolic_link="${link_dir}/J1_${J1}_J2_${J2}_J3_${J3}_lt_${LT}"
    [ ! -d "$link_dir" ] && mkdir -p "$link_dir" && echo "Created link directory $link_dir"

    # Clear out any old links/data
    echo "Clearing out $symbolic_link:"
    rm -rf "${symbolic_link:?}/radam/"*

    echo "Symbolic link for MG1D model: $symbolic_link"
    echo "Project directory: $project_dir"

    cd "$project_dir/python/rmsKit" || return

    # CPU configuration
    if [ "$n_cpu" -gt 0 ]; then
        echo "Using $n_cpu CPUs"
        export OMP_NUM_THREADS=$n_cpu
        export MKL_NUM_THREADS=$n_cpu
    else
        echo "Using all available CPUs"
    fi

    # Environment
    source /opt/materiapps/intel/env.sh
    source ~/worms/myenv/bin/activate

    # 1) Optimization
    python -u optimize_loc.py \
        -m $model_name \
        -lr 0.01 \
        -e $EPOCH \
        -M $M \
        -lt $LT \
        -J1 $J1 \
        -J2 $J2 \
        -J3 $J3 \
        -n $n_cpu \
        --loss none \
        --symbolic_link $symbolic_link \
        --stdout >> $log_file

    echo "Finished optimization for MG1D model with J1=${J1}, J2=${J2}, J3=${J3} on $n_cpu CPUs"

    # 2) Worm sampling (match BLBQ order and flags)
    # python -u -m run_worm \
    #     -m $model_name \
    #     --path $symbolic_link \
    #     -s $SWEEPS \
    #     -n $n_cpu \
    #     -k $M \
    #     --obc \
    #     --stdout >> $log_file

    python -u -m run_worm \
        -m $model_name \
        --path $symbolic_link \
        -s $SWEEPS \
        --original \
        -n $n_cpu \
        --obc \
        --stdout >> $log_file

    echo "Finished MG1D sampling for J1=${J1}, J2=${J2}, J3=${J3} on $n_cpu CPUs"

    # Optional: cleanup symlink
    echo "Cleaning up existing symbolic link: $symbolic_link"
    # [ -L "$symbolic_link" ] && unlink "$symbolic_link"
}

# Initialization
calculate_total_jobs
echo_jobs
echo "Sourcing script mg1d.sh"
export -f run_job
export -f calculate_parameters
export -f calculate_total_jobs
