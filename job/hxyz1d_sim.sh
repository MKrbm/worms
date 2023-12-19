#!/bin/bash

# n: Calculate the total number of jobs for HXYZ1D model

calculate_total_jobs() {
    Jx_values=($(seq -2 0.4 2))
    Jy_values=($(seq -2 0.4 2))
    Jz_values=(1)  # Only one Jz value in this case
    H_values=(0 0.5 1)

    num_Jx=${#Jx_values[@]}
    num_Jy=${#Jy_values[@]}
    num_Jz=${#Jz_values[@]}
    num_H=${#H_values[@]}

    total_jobs=$((num_Jx * num_Jy * num_Jz * num_H))
}

# Function to calculate Jx, Jy, Jz, and H from a single integer input
calculate_parameters() {
    calculate_total_jobs
    local task_id=$1

    i=$((task_id / (num_Jy * num_Jz * num_H)))
    j=$((task_id % (num_Jy * num_Jz * num_H) / (num_Jz * num_H)))
    z=$((task_id % (num_Jz * num_H) / num_H))
    k=$((task_id % num_H))

    Jx=$(printf "%.3f" "${Jx_values[$i]}")
    Jy=$(printf "%.3f" "${Jy_values[$j]}")
    Jz=$(printf "%.3f" "${Jz_values[$z]}")
    H=$(printf "%.3f" "${H_values[$k]}")
}

# Echo all jobs to confirm the parameters
echo_jobs() {

    for i in $(seq 0 $((total_jobs - 1))); do
        calculate_parameters "$i"
        echo "Jx=${Jx}, Jy=${Jy}, Jz=${Jz}, H=${H}"
    done
    echo "Total number of jobs for HXYZ1D model: $total_jobs"
}

# Core job function for HXYZ1D model
run_job() {
    local task_id=$1
    local project_dir=$2
    local n_cpu=$3

    calculate_parameters "$task_id"

    LT=1
    SWEEPS=1000000
    EPOCH=2000
    LR=0.005
    M=20
    model_name="HXYZ1D"
    log_file="${project_dir}/job/worm/${model_name}_Jx_${Jx}_Jy_${Jy}_Jz_${Jz}_H_${H}_output.log"
    link_dir="${project_dir}/job/link/${model_name}"
    symbolic_link="${link_dir}/Jx_${Jx}_Jy_${Jy}_Jz_${Jz}_hx_${H}_hz_0_lt_${LT}"

    # Create the link directory if it does not exist
    [ ! -d "$link_dir" ] && mkdir -p "$link_dir" && echo "Created link directory $link_dir"

    echo "Symbolic link for HXYZ1D model: $symbolic_link"
    echo "Project directory: $project_dir"

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

    # Running the simulation for HXYZ1D
    python -u optimize_loc.py -m $model_name -o Adam -lr $LR -e $EPOCH -M $M -lt $LT \
        -Jz "$Jz" -Jx "$Jx" -Jy "$Jy" -hx "$H" -n "$n_cpu" --symoblic_link "$symbolic_link" \
        --stdout >> "$log_file"

    echo "Finished optimization for HXYZ1D model with Jx=${Jx}, Jy=${Jy}, Jz=${Jz}, H=${H} in CPU ${n_cpu}"

	python -u -m run_worm -m HXYZ1D \
		-f "$symbolic_link" -s $SWEEPS --original -n "$n_cpu" \
		>> "$project_dir"/job/worm/Jx_"${Jx}"_Jy_"${Jy}"_Jz_"${Jz}"_H_"${H}"_output.log

	python -u -m run_worm -m HXYZ1D \
		-f "$symbolic_link" -s $SWEEPS -n "$n_cpu" \
		>> "$project_dir"/job/worm/Jx_"${Jx}"_Jy_"${Jy}"_Jz_"${Jz}"_H_"${H}"_output.log

    echo "Finished HXYZ1D model job with Jx=${Jx}, Jy=${Jy}, Jz=${Jz}, H=${H} in CPU ${n_cpu}"


    echo "Cleaning up existing symbolic links $symbolic_link"
    # Check if symbolic link already exists and remove it if it does
    # [ -L "$symbolic_link" ] && unlink "$symbolic_link"

}

# Initialization
calculate_total_jobs
echo_jobs
echo "Sourcing script hxyz1d.sh"
export -f run_job
export -f calculate_parameters
export -f calculate_total_jobs
