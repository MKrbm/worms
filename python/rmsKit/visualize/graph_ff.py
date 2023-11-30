import sys
import os
# Get the path of the current script's directory.
current_script_path = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to the sys.path.
parent_directory = os.path.dirname(current_script_path)
sys.path.append(parent_directory)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import get_seed_and_loss


N = 5000000
beta = 20


model_name = "FF2D_quetta"
df = get_seed_and_loss(f"../../worm_result/{model_name}")
df = df[df.sweeps == N]
model_name = "FF2D_zetta"
df1 = get_seed_and_loss(f"../../worm_result/{model_name}")
df1 = df1[df1.sweeps == 4999872]
df = pd.concat([df, df1])
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

        data_size = 0
        for key, _df in df_dict.items():
            df_dict[key] = _df.loc[_df.groupby(
                "temperature")["as"].idxmax()].sort_values(by="temperature")
            data_size += len(df_dict[key])

        if (data_size / len(df_dict)) < 4:
            print(
                "number of data is not sufficient for seed = {} and L = {}".format(seed, L))
            continue

        label_dict["mel"] = f'mel / loss = {df_dict["mel"].loss.unique()[0]}'
        label_dict["none"] = "none"

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
        ax1.set_title('Energy per site vs Beta at L = {}'.format(L))
        ax1.set_ylabel('energy')
        try:
            ax1.set_ylim(lower_bound, upper_bound)
        except ValueError:
            print("lower_bound / upper_bound was not valid (nan / inf))")
            continue

        ax1.legend()

        # Plotting AS vs Inverse Temperature in ax2
        ax2.errorbar(1/df_dict["mel"]['temperature'], df_dict["mel"]['as'],
                     yerr=df_dict["mel"]['as_error'], fmt='o-', capsize=5, label=label_dict["mel"])
        ax2.errorbar(1/df_dict["none"]['temperature'], df_dict["none"]['as'],
                     yerr=df_dict["none"]['as_error'], fmt='s-', capsize=5, label=label_dict["none"], alpha=0.7)
        ax2.set_title('Average Sign vs Beta at L = {}'.format(L))
        ax2.set_ylabel('Average Sign')
        ax2.legend()

        # Plotting AS vs Inverse Temperature (Log Scale) in ax3
        ax3.errorbar(1/df_dict["mel"]['temperature'], df_dict["mel"]['as'],
                     yerr=df_dict["mel"]['as_error'], fmt='o-', capsize=5, label=label_dict["mel"])
        ax3.errorbar(1/df_dict["none"]['temperature'], df_dict["none"]['as'],
                     yerr=df_dict["none"]['as_error'], fmt='s-', capsize=5, label=label_dict["none"], alpha=0.7)
        ax3.set_title('Average Sign vs Beta (log scale) at L = {}'.format(L))
        ax3.set_xlabel('Inverse Temperature (Beta)')
        ax3.set_ylabel('Average Sign')
        ax3.set_yscale('log')
        ax3.legend()

        fig.tight_layout()

        if not os.path.exists(f"image/{model_name}/N_{N:.0e}/comp_T"):
            os.makedirs(f"image/{model_name}/N_{N:.0e}/comp_T")
        fig.savefig(
            f"image/{model_name}/N_{N:.0e}/comp_T/L_{L}_r_{seed}.png")

        plt.close()

for t in sorted(df.temperature.unique()):
    df_t = df[df.temperature == t]
    seed_list = sorted(df_t.seed.unique())
    for seed in seed_list:
        label_dict = {}
        dfs = df_t[df_t.seed == seed]
        df_dict = {}
        df_dict["mel"] = dfs[dfs.u_path.str.contains("mel")]
        df_dict["none"] = dfs[dfs.u_path == ""]

        data_size = 0
        for key, _df in df_dict.items():

            df_dict[key] = _df.loc[_df.groupby(
                "n_sites")["as"].idxmax()].sort_values(by="n_sites")
            data_size += len(df_dict[key])

        if (data_size / len(df_dict)) < 4:
            print(
                "number of data is not sufficient for seed = {} and T = {}".format(seed, t))
            continue

        label_dict["mel"] = f'mel / loss = {df_dict["mel"].loss.unique()[0]}'
        label_dict["none"] = "none"

        fig, ax = plt.subplots(3, figsize=(10, 12))
        # gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.4)
        ax1 = ax[0]
        ax2 = ax[1]
        ax3 = ax[2]

        data = np.concatenate([df_dict["mel"]['e'], df_dict["none"]['e']])
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        print(Q1, Q3, f"seed: {seed} T: {t}")
        IQR = Q3 - Q1
        lower_bound = Q1 - 0.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Plotting Energy vs Inverse Temperature (Beta) in ax1
        ax1.errorbar(df_dict["mel"]['n_sites'], df_dict["mel"]['e'],
                     yerr=df_dict["mel"]['e_error'], fmt='o-', capsize=5, label=label_dict["mel"])
        ax1.errorbar(df_dict["none"]['n_sites'], df_dict["none"]['e'],
                     yerr=df_dict["none"]['e_error'], fmt='s-', capsize=5, label=label_dict["none"], alpha=0.7)

        ax1.set_title('Energy per site vs L at beta = {}'.format(t))

        ax1.set_ylabel('Energy')
        try:
            ax1.set_ylim(lower_bound, upper_bound)
        except ValueError:
            print("lower_bound / upper_bound was not valid (nan / inf))")
            continue
        ax1.legend()

        # Plotting AS vs Inverse Temperature in ax2

        ax2.errorbar(df_dict["mel"]['n_sites'], df_dict["mel"]['as'],
                     yerr=df_dict["mel"]['as_error'], fmt='o-', capsize=5, label=label_dict["mel"])
        ax2.errorbar(df_dict["none"]['n_sites'], df_dict["none"]['as'],
                     yerr=df_dict["none"]['as_error'], fmt='s-', capsize=5, label=label_dict["none"], alpha=0.7)

        ax2.set_title('Average Sign vs L at beta = {}'.format(t))
        ax2.set_ylabel('Average Sign')
        ax2.legend()

        # Plotting AS vs Inverse Temperature (Log Scale) in ax3
        ax3.errorbar(df_dict["mel"]['n_sites'], df_dict["mel"]['as'],
                     yerr=df_dict["mel"]['as_error'], fmt='o-', capsize=5, label=label_dict["mel"])
        ax3.errorbar(df_dict["none"]['n_sites'], df_dict["none"]['as'],
                     yerr=df_dict["none"]['as_error'], fmt='s-', capsize=5, label=label_dict["none"], alpha=0.7)

        ax3.set_title('Average Sign vs L (log scale) at beta = {}'.format(t))
        ax3.set_xlabel('L')
        ax3.set_ylabel('Average Sign')
        ax3.set_yscale('log')
        ax3.legend()

        fig.tight_layout()

        if not os.path.exists(f"image/{model_name}/N_{N:.0e}/comp_L"):
            os.makedirs(f"image/{model_name}/N_{N:.0e}/comp_L")
        fig.savefig(
            f"image/{model_name}/N_{N:.0e}/comp_L/T_{t}_r_{seed}.png")

        plt.close()
