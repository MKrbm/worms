#!/bin/bash

# Calculate the total number of jobs
# Calculate the total number of jobs
calculate_total_jobs() {
    Jx_values=($(seq -2 0.4 2))
    Jy_values=($(seq -2 0.4 2))
    Jz_values=(1)  # Define Jz values
    H_values=(0 0.5 1)

    num_Jx=${#Jx_values[@]}
    num_Jy=${#Jy_values[@]}
    num_Jz=${#Jz_values[@]}  # Number of Jz values
    num_H=${#H_values[@]}

    total_jobs=$((num_Jx * num_Jy * num_Jz * num_H))  # Updated total jobs calculation
}

# Function to calculate Jx, Jy, Jz, and H from a single integer input
calculate_parameters() {
    calculate_total_jobs
    local task_id=$1

    i=$((task_id / (num_Jy * num_Jz * num_H)))
    j=$((task_id % (num_Jy * num_Jz * num_H) / (num_Jz * num_H)))
    z=$((task_id % (num_Jz * num_H) / num_H))
    k=$((task_id % num_H))

    # Round off to three decimal places
    Jx=$(printf "%.3f" "${Jx_values[$i]}")
    Jy=$(printf "%.3f" "${Jy_values[$j]}")
    Jz=$(printf "%.3f" "${Jz_values[$z]}")
		H=$(printf "%.3f" "${H_values[$k]}")

		echo "Jx=$Jx, Jy=$Jy, Jz=$Jz, H=$H"
}

echo_total_jobs() {
    echo "Total number of jobs: $total_jobs"
}

# Core job function
run_job() {
		

	local task_id=$1
	local project_dir=$2
	local n_cpu=$3

	calculate_parameters $task_id

	LT=1
	SWEEPS=1000000
	EPOCH=2000
	LR=0.005
	M=2
	symoblic_link="${project_dir}/job/link/Jx_${Jx}_Jy_${Jy}_Jz_${Jz}_hx_${H}_hz_0_lt_${LT}"
	echo "symoblic_link: $symoblic_link"
	echo "project_dir: $project_dir"


	cd "$project_dir/python/rmsKit" || return

	# Add your environment setup if necessary
	#export OMP_NUM_THREADS=$n_cpu
	#export MKL_NUM_THREADS=$n_cpu

	source /opt/materiapps-gcc/env.sh
	source ~/worms/myenv/bin/activate

	python -u new_optimize_loc.py -m HXYZ1D \
		-loss mel -o Adam --lattice_type $LT -M $M -e $EPOCH -lr $LR -Jz $Jz -Jx $Jx -Jy $Jy -hx $H -n $n_cpu \
		--symoblic_link $symoblic_link \
		--stdout >> $project_dir/job/worm/Jx_${Jx}_Jy_${Jy}_Jz_${Jz}_H_${H}_output.log

	python -u -m run_worm -m HXYZ1D \
		-f $symoblic_link -s $SWEEPS --original -n $n_cpu \
		>> $project_dir/job/worm/Jx_${Jx}_Jy_${Jy}_Jz_${Jz}_H_${H}_output.log

	python -u -m run_worm -m HXYZ1D \
		-f $symoblic_link -s $SWEEPS -n $n_cpu \
		>> $project_dir/job/worm/Jx_${Jx}_Jy_${Jy}_Jz_${Jz}_H_${H}_output.log
	
	echo "Finish job with Jx=$Jx, Jy=$Jy, Jz=$Jz and H=$H in CPU $n_cpu"

	deactivate

}

# echo_total_jobs
echo "Sourcing script hxzy1d_sim.sh"
calculate_total_jobs
echo_total_jobs
export -f run_job 
export -f calculate_parameters
export -f calculate_total_jobs
