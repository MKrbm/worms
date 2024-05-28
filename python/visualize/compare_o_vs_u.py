import torch
import pandas as pd
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
import logging

PYTHON_DIR = Path(__file__).parents[1]
sys.path.insert(0, PYTHON_DIR.as_posix())
from rmsKit.utils.logdata_handler import parse_log_file

BASE_DIR = "/Users/keisuke/Documents/projects/todo/worms/job/FF1D_sps_8/FF1D"
FIGURE_DIR = PYTHON_DIR / "visualize" / "plots" / "compare_o_vs_u" / "sps_8"
Path(FIGURE_DIR).mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

seeds = [i for i in range(100, 1000, 1)]

for seed in seeds:
    dfo, metao = parse_log_file(f"{BASE_DIR}/sps_8_seed_{seed}_orth.log")
    dfu, metau = parse_log_file(f"{BASE_DIR}/sps_8_seed_{seed}_uni.log")

    assert metao.seed == metau.seed
    assert metao.model == metau.model

    orthogonal_loss_min = dfo["Best Loss at Iteration"].min()
    unitary_loss_min = dfu["Best Loss at Iteration"].min()

    if unitary_loss_min < orthogonal_loss_min:
        print(f"Seed {seed} has a better unitary loss: {unitary_loss_min:.4f} < {orthogonal_loss_min:.4f}")