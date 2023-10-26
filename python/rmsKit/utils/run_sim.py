import subprocess
import os
from datetime import datetime
from typing import Union


def run_ff_1d(
    seed: int,
    rank: int,
    sps: int,
    us: int,
    dimension: int = 1,
    M: int = 100,
    e: int = 1000,
    output_dir: str = "out/run_sim/",
) -> Union[subprocess.CompletedProcess, None]:
    # Set your parameters
    params = {
        "sps": sps,
        "rank": rank,
        "dimension": dimension,  # Currently, only 1D is available
        "us": us,
        "seed": seed,
        "M": M,
        "e": e,
    }

    # check if executable exists
    if not os.path.isfile("torch_optimize_loc.py"):
        print("torch_optimize_loc.py not found. Please check the path.")
        return None

    # Construct the command to run the optimization script
    command = f"python torch_optimize_loc.py -m FF1D -loss mel -o LION -e {params['e']} -M {params['M']} -lr 0.005 -r {params['rank']} --seed {params['seed']}"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construct the output file path
    output_filename = (
        f"output_seed_{params['seed']}_rank_{params['rank']}_sps_{params['sps']}_us_{params['us']}_{timestamp}.txt"
    )
    output_filepath = os.path.join(output_dir, output_filename)

    # Run the command
    with open(output_filepath, "w") as output_file:
        process = subprocess.run(command, shell=True, stdout=output_file, stderr=subprocess.STDOUT, text=True)

    # Check if the process ran successfully
    if process.returncode == 0:
        print(f"Optimization for seed {seed} ran successfully. Results written to {output_filepath}")
    else:
        print(f"Optimization for seed {seed} failed. See {output_filepath} for error details.")

    return process
