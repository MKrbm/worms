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

BASE_DIR = "/Users/keisuke/Documents/projects/todo/worms/job/FF1D"
FIGURE_DIR = PYTHON_DIR / "visualize" / "figs"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

seeds = [i for i in range(0, 24, 1)]

for seed in seeds:
    logger.info(f"Processing seed {seed}")
    dfo, metao = parse_log_file(f"{BASE_DIR}/seed_{seed}_orth.log")
    dfu, metau = parse_log_file(f"{BASE_DIR}/seed_{seed}_uni.log")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot 1: Distribution of Starting Loss Values (orthogonal)
    dfo["Loss at Epoch Start"].hist(bins=20, ax=axes[0, 0])
    axes[0, 0].set_xlabel("Loss at Epoch Start")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Distribution of Starting Loss Values (orthogonal)")

    # Plot 2: Distribution of Starting Loss Values (unitary)
    dfu["Loss at Epoch Start"].hist(bins=20, ax=axes[0, 1])
    axes[0, 1].set_xlabel("Loss at Epoch Start")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Distribution of Starting Loss Values (unitary)")

    # Plot 3: Distribution of Best Loss Values (orthogonal)
    dfo["Best Loss at Iteration"].hist(bins=20, ax=axes[1, 0])
    axes[1, 0].set_xlabel("Best Loss at Iteration")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Distribution of Best Loss Values (orthogonal)")

    # Plot 4: Distribution of Best Loss Values (unitary)
    dfu["Best Loss at Iteration"].hist(bins=20, ax=axes[1, 1])
    axes[1, 1].set_xlabel("Best Loss at Iteration")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Distribution of Best Loss Values (unitary)")

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / f"seed_{seed}_loss_distributions.png")
    plt.close(fig)
    # plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))

    unitary_loss_min = dfu["Best Loss at Iteration"].min()
    orthogonal_loss_min = dfo["Best Loss at Iteration"].min()

    combined_data = [dfo["Best Loss at Iteration"], dfu["Best Loss at Iteration"]]
    box = ax.boxplot(
        combined_data,
        tick_labels=[
            f"orthogonal (min: {orthogonal_loss_min:.2f})",
            f"unitary (min: {unitary_loss_min:.2f})",
        ],
        patch_artist=True,
    )

    # Adding colors to box plots
    colors = ["lightblue", "lightgreen"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)

    # Adding explanation of the box plot
    for i, data in enumerate(combined_data, start=1):
        median = np.median(data)
        ax.text(
            i,
            median,
            f"Median: {median:.2f}",
            horizontalalignment="center",
            verticalalignment="bottom",
            color="black",
        )

    ax.set_xlabel("Log Files")
    ax.set_ylabel("Best Loss at Iteration")
    ax.set_title("Boxplot of Best Loss at Iteration for orthogonal vs unitary")

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / f"seed_{seed}_boxplot_best_loss.png")
    # plt.show()
    plt.close(fig)
