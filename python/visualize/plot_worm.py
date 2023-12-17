"""Visualize the result of worm simulation.

Plot the energy and average sign as a function of inverse temperature (beta) and lattice size (L).
"""
import sys
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
sys.path.append(Path(__file__).resolve().parent.parent.as_posix())
from rmsKit.utils import result_to_dataframe  # noqa: E402


def extract_parameters_from_path(path: str) -> Dict[str, float]:
    # Split the path and get the relevant part with parameters
    parts = path.split('/')
    params_part = parts[-3]  # The third from the last part contains the parameters

    params_split = params_part.split('_')
    if len(params_split) % 2 != 1:
        raise ValueError("""
        The format of the path name is : modelname_param1_value1_param2_value2_...
        However, the number of parameters is even. Which means that there is a parameter without a value.
        """)

    model_name = params_split[0]
    params_split = params_split[1:]
    params = {
        model_name: model_name
    }
    for i in range(0, len(params_split), 2):
        try:
            params[params_split[i]] = float(params_split[i + 1])
        except ValueError as e:
            print(e)
            raise ValueError(
                "The parameter value is not a float: {}".format(params_split[i + 1]))


IMAGE_PATH = Path("visualize/image")
WORM_RESULT_PATH = Path("../output_worm/zetta")
MODEL_NAME = "HXYZ1D"
N = 1000000
BETA_THRES = 20
if not IMAGE_PATH.exists():
    raise FileNotFoundError("{} does not exist.".format(IMAGE_PATH.resolve()))
if not WORM_RESULT_PATH.exists():
    raise FileNotFoundError("{} does not exist.".format(WORM_RESULT_PATH.resolve()))

image_model_dir = IMAGE_PATH / MODEL_NAME
worm_result_path = WORM_RESULT_PATH / MODEL_NAME

print("looking for the result in {}".format(worm_result_path.resolve()))
print(worm_result_path.resolve())
df = result_to_dataframe(worm_result_path.resolve().as_posix())
df = df[df.sweeps == N]

df = df[df["temperature"] >= 1 / BETA_THRES]
print("temeprature simulated: {}".format(np.sort(df.temperature.unique())))
print("L simulated: {}".format(np.sort(df.n_sites.unique())))
