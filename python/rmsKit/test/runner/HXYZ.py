from . import run
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd


def run_HXYZ(params: Dict,
             rmsKit_directory,
             output_dir) -> Tuple[pd.DataFrame,
                                  pd.DataFrame,
                                  pd.DataFrame,
                                  pd.DataFrame]:

    output_dir = Path(output_dir)
    model_name = "HXYZ1D" if "L2" not in params else "HXYZ2D"
    cmd = [
        "python",
        "solver_torch.py",
        "-m",
        model_name,
        "-L1",
        str(params["L1"]),
        "-L2" if "L2" in params else "",
        str(params["L2"]) if "L2" in params else "",
        "-Jz",
        str(params["Jz"]),
        "-Jx",
        str(params["Jx"]),
        "-Jy",
        str(params["Jy"]),
        "-hx",
        str(params["hx"]),
        "-hz",
        str(params["hz"]),
        "--stdout",
    ]

    cmd_solver = " ".join(cmd)

    cmd = [
        "python",
        "optimize_loc.py",
        "-m",
        model_name,
        "-Jz",
        str(params["Jz"]),
        "-Jx",
        str(params["Jx"]),
        "-Jy",
        str(params["Jy"]),
        "-hx",
        str(params["hx"]),
        "-hz",
        str(params["hz"]),
        "-e 1000",
        "-M 10",
        "--stdout",
    ]

    cmd_optimize = " ".join(cmd)

    return run.run(
        params,
        rmsKit_directory,
        output_dir,
        model_name,
        cmd_solver,
        cmd_optimize,
        N=10**6,
        beta=[0.1, 0.5, 1])
