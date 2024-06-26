import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import path_with_lowest_loss, sum_ham  # noqa: E402
import numpy as np  # noqa: E402
from lattice import FF  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
# Assuming the 'lattice' module provides the FF.block1D function
# Assuming the 'utils' module provides the path_with_lowest_loss and sum_ham functions


def check_valid_seeds(base_path, sps, lt, setting_name, range_seed=range(3000, 4000)):
    valid_seeds = []
    valid_path = []
    for seed in range_seed:
        dir_path = f"{base_path}/s_{sps}_r_2_lt_{lt}_d_1_seed_{seed}/{setting_name}"
        if os.path.isdir(dir_path):
            valid_seeds.append(seed)
            valid_path.append(dir_path)
    return valid_seeds, valid_path


def load_matrix(base_dir, file_name):
    return np.load(os.path.join(base_dir, file_name))


def compute_gap_sys(h, L, sps):
    H = sum_ham(h, [[i, (i + 1) % L] for i in range(L)], L, sps)
    E = np.linalg.eigvalsh(H)
    return E[1] - E[0]


def compute_gap(A):
    A_ = A.transpose(1, 0, 2)
    A_tilde = np.einsum("ijk,ilm->jlkm", A_, A_).reshape(4, 4)
    eigenvalues = np.linalg.eigvals(A_tilde)
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    return sorted_eigenvalues.real[0] - sorted_eigenvalues.real[1]


def get_projector(A, bd, sps):
    A2 = np.einsum("ijk,klm->jlim", A, A).reshape(sps**2, bd**2)
    U, s, V = np.linalg.svd(A2)
    Up = U[:, len(s):]
    h = Up @ Up.T
    # Additional logic to compute max_gap if necessary
    return h, 0  # Assuming max_gap is 0 for now


def compute_results(seed, sps, bd, lt, base_dir, L):

    u_path, lowest_val = path_with_lowest_loss(base_dir)
    u = load_matrix(u_path, "0.npy")
    h_matrix = -load_matrix(base_dir, "H/0.npy")
    U = np.kron(u, u)
    hu = U @ h_matrix @ U.T

    # sps will be a power of lt, and here calculate the original sps
    sps = np.round(sps ** (1 / lt)).astype(int)
    A = FF.block1D(bd, sps, bd, seed=seed)
    gap_transfer = compute_gap(A)
    h, gap_loc = get_projector(A, bd, sps)
    gap_sys = compute_gap_sys(h, L, sps)
    return {
        "seed": seed,
        "loss": lowest_val,
        "gap_loc": gap_loc,
        "gap_sys": gap_sys,
        "gap_transfer": gap_transfer,
        "n_0": np.sum(np.round(hu, 5) == 0)
    }


def main():
    bd = 2
    sps = 8
    lt = 1
    sps = sps ** lt
    res = []
    # base_path = "array/torch/FF1D_loc_lt_3"
    base_path = "array/torch/FF1D_8"
    # base_path = "array/torch/FF1D_loc_lt_2"
    setting_name = f"{lt}_mel_Adam/lr_0.001_epoch_4000"
    # setting_name = "2_mel_Adam/lr_0.001_epoch_8000"
    # base_path = "array/torch/FF1D_lt_2"

    valid_seeds, valid_path = check_valid_seeds(
        base_path, sps, lt=lt, setting_name=setting_name, range_seed=range(3000, 4000))

    for seed, array_dir in zip(valid_seeds, valid_path):
        try:
            result = compute_results(seed, sps, bd, lt, array_dir, 3)
            res.append(result)
            print(f"finish seed = {seed}")
        except Exception as e:
            print(f"error seed = {seed}")
            print(e)

    df = pd.DataFrame(res)
    df.loss = df.loss / lt
    df["log_gap_loc"] = df['gap_loc'].apply(np.log)
    df["log_gap_sys"] = df['gap_sys'].apply(np.log)
    df["log_loss"] = df['loss'].apply(lambda x: np.log(x+0.0001))

    # Create a pairplot to visualize the correlations between loss, gap, and minimum energy.
    sns.pairplot(
        df[['loss', 'gap_transfer', "log_gap_sys", "n_0"]], kind="hist")

    # Set titles and labels using a variable.
    plot_title = f'Correlation between Loss, gap_transfer, gap_sys, n==0 for SPS={sps} / bd={bd} / lt={lt}'
    plt.suptitle(plot_title, size=10, y=1.05)  # Adjust y for title position

    # Save the plot with a variable filename.
    plot_filename = f'visualize/image/FF1D/pairplot_sps_{sps}_bd_{bd}_lt_{lt}.png'
    plt.savefig(plot_filename, dpi=300)

    fig, ax = plt.subplots(2, 5, figsize=(20, 10))  # Corrected order here
    for i in range(10):
        sample_df = df.iloc[i*100:(i+1)*100]
        ax[i//5, i % 5].scatter(sample_df['log_gap_sys'], sample_df['loss'])
        ax[i//5, i % 5].set_title(f"seed = {sample_df['seed'].iloc[0]}")
        for j in range(len(sample_df)):  # Changed to 'j' to avoid conflict with outer loop
            ax[i//5, i % 5].annotate(sample_df['seed'].iloc[j],  # Also changed to 'j' here
                                     (sample_df['log_gap_sys'].iloc[j],
                                      sample_df['loss'].iloc[j]),
                                     textcoords="offset points", xytext=(0, 10), ha='center')

    fig.suptitle(
        f'Correlation between Loss and gap_sys for SPS={sps} / bd={bd} / lt={lt}', size=10, y=1.05)
    fig.tight_layout()
    plot_filename = f'visualize/image/FF1D/scatter_sps_{sps}_bd_{bd}_lt_{lt}.png'
    plt.savefig(plot_filename, dpi=300)


if __name__ == "__main__":
    main()
