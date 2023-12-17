from . import run
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd


def run_HXYZ1D(params: Dict,
               rmsKit_directory,
               output_dir) -> Tuple[pd.DataFrame,
                                    pd.DataFrame,
                                    pd.DataFrame,
                                    pd.DataFrame]:

    output_dir = Path(output_dir)
    cmd = [
        "python",
        "solver_torch.py",
        "-m",
        "HXYZ1D",
        "-L1",
        str(params["L1"]),
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
        "HXYZ1D",
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

    return run.run(params, rmsKit_directory, output_dir, "HXYZ1D" , cmd_solver, cmd_optimize)
