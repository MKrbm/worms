import matplotlib.pyplot as plt
import sys
import os
import numpy as np

# Get the path of the current script's directory.
current_script_path = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to the sys.path.
parent_directory = os.path.dirname(current_script_path)
sys.path.append(parent_directory)

from utils import get_seed_and_loss


N = 4*10**6
beta = 20


df = get_seed_and_loss("../../../worm_result")
df = df[df.sweeps == N]
print(df.temperature)
df = df[df["temperature"] >= 1 / beta]

for L in sorted(df.n_sites.unique()):
    df_n = df[df.n_sites == L]
    seed_list = sorted(df_n.seed.unique())
    for seed in seed_list:
        label_dict = {}
        dfs = df_n[df_n.seed == seed]
        df_dict = {}
        df_dict["mel"] = dfs[dfs.u_path.str.contains("mel")]
        df_dict["none"] = dfs[dfs.u_path == ""]

        label_dict["mel"] = f'mel / loss = {df_dict["mel"].loss.unique()[0]}'
        label_dict["none"] = "none"

        data_size = 0
        for key, _df in df_dict.items():
            df_dict[key] = _df.loc[_df.groupby(
                "temperature")["as"].idxmax()].sort_values(by="temperature")
            data_size += len(df_dict[key])

        if (data_size / len(df_dict)) < 4:
            print("number of data is not sufficient")
            continue

        fig, ax = plt.subplots(3, figsize=(10, 12))
        # gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.4)
        ax1 = ax[0]
        ax2 = ax[1]
        ax3 = ax[2]

        data = np.concatenate([df_dict["mel"]['e'], df_dict["none"]['e']])
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        print(Q1, Q3, f"seed: {seed} L: {L}")
        IQR = Q3 - Q1
        lower_bound = Q1 - 0.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Plotting Energy vs Inverse Temperature (Beta) in ax1
        ax1.errorbar(1/df_dict["mel"]['temperature'], df_dict["mel"]['e'],
                     yerr=df_dict["mel"]['e_error'], fmt='o-', capsize=5, label=label_dict["mel"])
        ax1.errorbar(1/df_dict["none"]['temperature'], df_dict["none"]['e'],
                     yerr=df_dict["none"]['e_error'], fmt='s-', capsize=5, label=label_dict["none"], alpha=0.7)
        ax1.set_title('Energy vs Beta')
        ax1.set_ylabel('Energy')
        ax1.set_ylim(lower_bound, upper_bound)
        ax1.legend()

        # Plotting AS vs Inverse Temperature in ax2
        ax2.errorbar(1/df_dict["mel"]['temperature'], df_dict["mel"]['as'],
                     yerr=df_dict["mel"]['as_error'], fmt='o-', capsize=5, label=label_dict["mel"])
        ax2.errorbar(1/df_dict["none"]['temperature'], df_dict["none"]['as'],
                     yerr=df_dict["none"]['as_error'], fmt='s-', capsize=5, label=label_dict["none"], alpha=0.7)
        ax2.set_title('Average Sign vs Beta')
        ax2.set_ylabel('Average Sign')
        ax2.legend()

        # Plotting AS vs Inverse Temperature (Log Scale) in ax3
        ax3.errorbar(1/df_dict["mel"]['temperature'], df_dict["mel"]['as'],
                     yerr=df_dict["mel"]['as_error'], fmt='o-', capsize=5, label=label_dict["mel"])
        ax3.errorbar(1/df_dict["none"]['temperature'], df_dict["none"]['as'],
                     yerr=df_dict["none"]['as_error'], fmt='s-', capsize=5, label=label_dict["none"], alpha=0.7)
        ax3.set_title('Average Sign vs Beta (log scale)')
        ax3.set_xlabel('Inverse Temperature (Beta)')
        ax3.set_ylabel('Average Sign')
        ax3.set_yscale('log')
        ax3.legend()

        fig.tight_layout()

        if not os.path.exists(f"image/FF1D/N_{N:.0e}/comp_T"):
            os.makedirs(f"image/FF1D/N_{N:.0e}/comp_T")
        fig.savefig(
            f"image/FF1D/N_{N:.0e}/comp_T/L_{L}_r_{seed}.png")

        plt.close()
