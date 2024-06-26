from . import run
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd


def run_MG1D(params: Dict,
             rmsKit_directory,
             output_dir) -> Tuple[pd.DataFrame,
                                  pd.DataFrame,
                                  pd.DataFrame,
                                  pd.DataFrame]:

    model_name = "MG1D"
    output_dir = Path(output_dir)
    cmd = [
        "python",
        "solver_torch.py",
        "-m",
        model_name,
        "-L1",
        str(params["L1"]),
        "-J1",
        str(params["J1"]),
        "-J2",
        str(params["J2"]),
        "-J3",
        str(params["J3"]),
        "-lt",
        str(params["lt"]),
        "--stdout",
    ]

    cmd_solver = " ".join(cmd)

    cmd = [
        "python",
        "optimize_loc.py",
        "-m",
        model_name,
        "-J1",
        str(params["J1"]),
        "-J2",
        str(params["J2"]),
        "-J3",
        str(params["J3"]),
        "-lt",
        str(params["lt"]),
        "--stdout",
        "-e 2000",
        "-M 20",
        "-lr 0.001",
        "-o Adam",
        "--stdout",
    ]
    cmd_optimize = " ".join(cmd)

    return run.run(params,
                   rmsKit_directory,
                   output_dir,
                   model_name,
                   cmd_solver,
                   cmd_optimize,
                   N=10**6,
                   beta=[0.1, 0.5, 1])
