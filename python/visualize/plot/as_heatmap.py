"""plot average sign and loss as heatmap."""
import numpy as np  # noqa
# import os  # noqa
# import matplotlib.gridspec as gridspec  # noqa
import matplotlib.pyplot as plt  # noqa
import pandas as pd  # noqa
import itertools  # noqa
# import pandas  # noqa

import sys  # noqa
from pathlib import Path  # noqa
PYTHON_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, PYTHON_DIR.resolve().as_posix())
from rmsKit import utils  # noqa
logger = utils.get_logger("log.log", level="INFO", stdout=True)
parser = utils.parser.get_parser()
args, _, _ = utils.parser.get_params_parser(parser)


IMAGE_PATH = PYTHON_DIR / "visualize" / "image"
WORM_RESULT_PATH = PYTHON_DIR / "rmsKit" / "array" / "quetta"

model_name = args.model
image_model_dir = IMAGE_PATH / model_name
worm_result_path = WORM_RESULT_PATH / (model_name+"_loc")

N = 10**6
BETA_THRES = 20

if not IMAGE_PATH.exists():
    raise FileNotFoundError("{} does not exist.".format(IMAGE_PATH.resolve()))
if not WORM_RESULT_PATH.exists():
    raise FileNotFoundError("{} does not exist.".format(WORM_RESULT_PATH.resolve()))
if not worm_result_path.exists():
    raise FileNotFoundError("{} does not exist.".format(worm_result_path.resolve()))

summary_files = utils.path.find_summary_files(worm_result_path)
df = utils.path.get_df_from_summary_files(summary_files, N)
df = df[df.sweeps == N]
df = df[df["T"] >= 1 / BETA_THRES]
logger.info("temeprature simulated: {}".format(np.sort(df["T"].unique())))
logger.info("L simulated: {}".format(np.sort(df.n_sites.unique())))

params_df = utils.param_dict_normalize(df['ham_path'].apply(utils.extract_parameters_from_path))
df = pd.concat([df, params_df], axis=1)
df = df.rename(columns={"T": "temperature"})
print(df.columns)

image_model_dir.mkdir(parents=False, exist_ok=True)

# Define the heatmap plotting function


# Define the heatmap plotting function
def plot_heatmap(df, fixed_params, heatmap_params, model_name, image_model_dir):
    """Plot the heatmap of the average sign and loss as a function of two parameters."""
    for fixed_values in itertools.product(*fixed_params.values()):
        filtered_df = df.copy()
        figure_name_parts = []
        # print(fixed_values)
        for key, value in zip(fixed_params.keys(), fixed_values):
            filtered_df = filtered_df[filtered_df[key] == value]
            # print(key, value, len(filtered_df))
            figure_name_parts.append(f"{key}_{value}")


        x_param, y_param = heatmap_params
        x_values = np.sort(filtered_df[x_param].unique())
        y_values = np.sort(filtered_df[y_param].unique())
        x, y = np.meshgrid(x_values, y_values)

        zs = {
            "NegativeSign (optimized)": [],
            "NegativeSign (initial)": [],
            "Loss (optimized)": [],
            "Loss (initial)": [],
        }

        # Process data for heatmap
        # print(x, y)
        for Jx, Jy in zip(x.reshape(-1), y.reshape(-1)):
            df_plot = filtered_df[(filtered_df[x_param] == Jx) & (filtered_df[y_param] == Jy)]
            df_u = df_plot[~df_plot.loss.isna()]
            df_h = df_plot[df_plot.loss.isna()]

            if len(df_u) == 0 or len(df_h) == 0:
                logger.debug(f"Skipping {x_param}={Jx}, {y_param}={Jy}")
                continue

            ah = df_h["as"].min()
            ah_err = df_h["as_error"].min() * np.sqrt(N)
            idx = np.argmin(df_u.loss.values)
            loss = df_u.loss.values[idx]
            init_loss = df_u.init_loss.values[idx]
            au = df_u["as"].values[idx]
            au_err = df_u["as_error"].values[idx] * np.sqrt(N)

            zs["NegativeSign (optimized)"].append(1 - au)
            zs["NegativeSign (initial)"].append(1 - ah)
            zs["Loss (optimized)"].append(loss)
            zs["Loss (initial)"].append(init_loss)

        # Plotting
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        max_loss = max(np.max(zs["Loss (optimized)"]), np.max(zs["Loss (initial)"]))
        # max_neg = max(np.max(zs["NegativeSign (optimized)"]), np.max(zs["NegativeSign (initial)"]))
        # max_neg = min(max_neg, 100)
        max_neg = 1
        if np.abs(max_loss) < 1e-5:
            max_loss = 1

        for i, (key, z) in enumerate(zs.items()):
            Z = np.array(z).reshape(x.shape)
            vmin, vmax = (0, max_loss) if "Loss" in key else (0, max_neg)
            c = ax[i // 2, i % 2].imshow(Z, cmap='RdPu', vmin=vmin, vmax=vmax, aspect="auto",
                                         extent=[x.min(), x.max(), y.min(), y.max()],
                                         origin='lower', interpolation='none')
            ax[i // 2, i % 2].set_title(key)
            ax[i // 2, i % 2].set_xlabel(x_param)
            ax[i // 2, i % 2].set_ylabel(y_param)
            fig.colorbar(c, ax=ax[i // 2, i % 2])

        plt.tight_layout()
        figure_name = f"AsHeatmap_{'_'.join(figure_name_parts)}.png"
        figure_path = image_model_dir / figure_name
        plt.savefig(figure_path, bbox_inches='tight')
        logger.info(f"Figure saved to {figure_path}")


if model_name == "HXYZ1D":
    fixed_params_HXYZ1D = {
        "temperature": np.sort(
            df.temperature.unique()), "n_sites": np.sort(
            df.n_sites.unique()), "hx": np.sort(
                df.hx.unique())}
    plot_heatmap(df, fixed_params_HXYZ1D, ('Jx', 'Jy'), model_name, image_model_dir)


if model_name == "HXYZ2D":
    fixed_params_HXYZ2D = {
        "temperature": np.sort(
            df.temperature.unique()), "n_sites": np.sort(
            df.n_sites.unique()), "hx": np.sort(
                df.hx.unique())}
    plot_heatmap(df, fixed_params_HXYZ2D, ('Jx', 'Jy'), model_name, image_model_dir)

elif model_name == "MG1D":
    fixed_params_MG1D = {
        "temperature": np.sort(df.temperature.unique()),
        "n_sites": np.sort(df.n_sites.unique()),
    }
    plot_heatmap(df, fixed_params_MG1D, ('J2', 'J3'), model_name, image_model_dir)

elif model_name == "BLBQ1D":
    fixed_params_BLBQ1D = {
        "temperature": np.sort(df.temperature.unique()),
        "n_sites": np.sort(df.n_sites.unique()),
        "J1": [1],
    }
    plot_heatmap(df, fixed_params_BLBQ1D, ('J0', 'hx'), model_name, image_model_dir)

elif model_name == "SS2D":
    fixed_params_MG1D = {
        "temperature": np.sort(
            df.temperature.unique()), "n_sites": np.sort(
            df.n_sites.unique()), "J0": [1]}
    plot_heatmap(df, fixed_params_MG1D, ('J1', 'J2'), model_name, image_model_dir)
