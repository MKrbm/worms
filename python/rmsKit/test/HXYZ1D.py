import pandas as pd
import re
import datetime
import contextlib
import logging
import subprocess
import os
import argparse
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List
from runner.HXYZ import run_HXYZ1D 
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.parser import get_parser  # noqa: E402

NOW = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = "out"



# Get the full path of the current file
current_file = Path(os.path.abspath(__file__))

# Extract the filename
filename = current_file.stem

# Extract the parent directory
# parent_directory = os.path.dirname(current_file)
# output_file = os.path.join(parent_directory, output_dir, f"{{}}_{NOW}_{filename}.txt")
output_file = (current_file.parent / output_dir / f"{{}}_{NOW}_{filename}.txt").as_posix()
rmsKit_directory = current_file.parent.parent.parent / "rmsKit"

args, params, hash_str = get_parser(length=True, model="HXYZ1D")
params["L1"] = args.length1

os.chdir(current_file.parent.parent)




if __name__ == "__main__":
    run_HXYZ1D(params, rmsKit_directory, output_file)
