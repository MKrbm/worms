#!/bin/bash
#
# ss2d_sim.sh  — Batch driver for the SS2D model
#   (structure and options harmonised with mg1d_sim.sh)

# ------------------------------------------------------------
# 1) Job-grid definition
calculate_total_jobs() {
    J0_values=(1)                       # Fixed
    J1_values=($(seq 0 0.1 2.0))
    J2_values=($(seq 0 0.1 2.0))

    num_J0=${#J0_values[@]}
    num_J1=${#J1_values[@]}
    num_J2=${#J2_values[@]}

    total_jobs=$((num_J0 * num_J1 * num_J2))
}

# ------------------------------------------------------------
# 2) Map Slurm-style $SLURM_ARRAY_TASK_ID → (J0,J1,J2)
calculate_parameters() {
    local task_id=$1

    i=$(( task_id / (num_J1 * num_J2) ))
    j=$(( task_id % (num_J1 * num_J2) / num_J2 ))
    k=$(( task_id % num_J2 ))

    J0=$(printf "%.3f" "${J0_values[$i]}")
    J1=$(printf "%.3f" "${J1_values[$j]}")
    J2=$(printf "%.3f" "${J2_values[$k]}")
}

# ------------------------------------------------------------
# 3) Debug helper: list every job
echo_jobs() {
    for idx in $(seq 0 $((total_jobs - 1))); do
        calculate_parameters "$idx"
        echo "J0=${J0}  J1=${J1}  J2=${J2}"
    done
    echo "Total number of jobs for SS2D model: $total_jobs"
}

# ------------------------------------------------------------
# 4) Main job runner — called by Slurm array
run_job() {
    local task_id=$1
    local project_dir=$2
    local n_cpu=$3

    calculate_parameters "$task_id"

    # ---------- optimisation / simulation constants ----------
    LT=1
    SWEEPS=100000              # match MG1D
    EPOCH=5000
    M=70                       # match MG1D
    LR=0.01                    # match MG1D
    model_name="SS2D"

    # ---------- logging & symlinks ----------
    log_dir="${project_dir}/job/log/${model_name}"
    mkdir -p "$log_dir"

    log_file="${log_dir}/J0_${J0}_J1_${J1}_J2_${J2}_lt_${LT}.log"

    link_dir="${project_dir}/job/link/${model_name}"
    symbolic_link="${link_dir}/J0_${J0}_J1_${J1}_J2_${J2}_lt_${LT}"
    mkdir -p "$link_dir"

    # wipe any stale optimiser state
    echo "Clearing out ${symbolic_link}/radam/"
    rm -rf "${symbolic_link:?}/radam/"*

    # ---------- environment ----------
    cd "${project_dir}/python/rmsKit" || exit 1

    if [[ $n_cpu -gt 0 ]]; then
        echo "Using $n_cpu CPUs"
        export OMP_NUM_THREADS=$n_cpu
        export MKL_NUM_THREADS=$n_cpu
    else
        echo "Using all available CPUs"
    fi

    source /opt/materiapps/intel/env.sh
    source ~/worms/myenv/bin/activate

    # ---------- 1) optimise local tensors ----------
    python -u optimize_loc.py \
        -m "$model_name" \
        -lr "$LR" \
        -e "$EPOCH" \
        -M "$M" \
        -lt "$LT" \
        -J0 "$J0" -J1 "$J1" -J2 "$J2" \
        -n "$n_cpu" \
        --symbolic_link "$symbolic_link" \
        --stdout >> "$log_file"

    echo "Finished optimisation for SS2D  (J0=${J0}, J1=${J1}, J2=${J2})" >> "$log_file"

    # ---------- 2) Worm sampling ----------
    # 2-a) optimised Hamiltonian first
    python -u -m run_worm \
        -m "$model_name" \
        --path "$symbolic_link" \
        -s "$SWEEPS" \
        -k 1 \
        -n "$n_cpu" \
        --stdout >> "$log_file"

    # 2-b) original Hamiltonian
    python -u -m run_worm \
        -m "$model_name" \
        --path "$symbolic_link" \
        -s "$SWEEPS" \
        --original \
        -k 1 \
        -n "$n_cpu" \
        --stdout >> "$log_file"

    echo "Completed SS2D sampling (optimised + original)" >> "$log_file"
}

# ------------------------------------------------------------
# 5) Export for use in sourcing context
calculate_total_jobs
echo_jobs
echo "Sourcing script ss2d_sim.sh"
export -f run_job
export -f calculate_parameters
export -f calculate_total_jobs
