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
python_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, python_dir.resolve().as_posix())
from rmsKit import utils  # noqa
logger = utils.get_logger("log.log", level="INFO", stdout=True)
parser = utils.parser.get_parser()
args, _, _ = utils.parser.get_params_parser(parser)

IMAGE_PATH = python_dir / "visualize" / "image"
WORM_RESULT_PATH = python_dir / "rmsKit" / "array" / "zetta"

model_name = args.model

N = 10**6
BETA_THRES = 20
if not IMAGE_PATH.exists():
    raise FileNotFoundError("{} does not exist.".format(IMAGE_PATH.resolve()))
if not WORM_RESULT_PATH.exists():
    raise FileNotFoundError("{} does not exist.".format(WORM_RESULT_PATH.resolve()))

image_model_dir = IMAGE_PATH / model_name
worm_result_path = WORM_RESULT_PATH / (model_name+"_loc")

if not worm_result_path.exists():
    raise FileNotFoundError("{} does not exist.".format(worm_result_path.resolve()))

summary_files = utils.path.find_summary_files(worm_result_path)
df = utils.path.get_df_from_summary_files(summary_files, N)
df = df[df.sweeps == N]
df = df[df["T"] >= 1 / BETA_THRES]
logger.info("temeprature simulated: {}".format(np.sort(df["T"].unique())))
logger.info("L simulated: {}".format(np.sort(df.n_sites.unique())))

params_df = utils.param_dict_normalize(df['ham_path'].apply(utils.extract_parameters_from_path))
df0 = pd.concat([df, params_df], axis=1)
df0 = df0.rename(columns={"T": "temp"})


if model_name == "HXYZ1D":

    h = 0.5
    T = 0.25
    L = 10
    df = df0
    hz_list = np.sort(df.hz.unique())
    hx_list = np.sort(df.hx.unique())
    T_list = np.sort(df.temp.unique())
    L_list = np.sort(df.n_sites.unique())

    for h, T, L in itertools.product(hx_list, T_list, L_list):

        # Determine a figure name that reflects the model and parameters
        figure_name = f"AsHeatmap_h_{h}_T_{T}_L_{L}.png"
        figure_path = image_model_dir / figure_name
        logger.info(f"Plotting {figure_name}")

        df = df0[(df0.temp == T) & (df0.n_sites == L) & (df0.hx == h)]

        Jx_list = np.sort(df.Jx.unique())
        Jy_list = np.sort(df.Jy.unique())
        Jz_list = np.sort(df.Jz.unique())

        zs = {
            "NegativeSign (optimized)": [],
            "NegativeSign (initial)": [],
            "Loss (optimized)": [],
            "Loss (initial)": [],
        }
        x, y = np.meshgrid(Jx_list, Jy_list)
        skip_flag = False
        for Jx, Jy in zip(x.reshape(-1), y.reshape(-1)):
            df_plot = df[df.hx == h]
            df_plot = df_plot[(df_plot.Jx == Jx) & (df_plot.Jy == Jy)]
            df_u = df_plot[~df_plot.loss.isna()]
            df_h = df_plot[df_plot.loss.isna()]
            if len(df_u) == 0 or len(df_h) == 0:
                logger.debug(f"Skipping Jx={Jx}, Jy={Jy}")
                skip_flag = True
                break
        #     au = df_u["as"].min()
            ah = df_h["as"].min()
            idx = np.argmin(df_u.loss.values)
            loss = df_u.loss.values[idx]
            init_loss = df_u.init_loss.values[idx]
            au = df_u["as"].values[idx]

            zs["NegativeSign (optimized)"].append(au)
            zs["NegativeSign (initial)"].append(ah)
            zs["Loss (optimized)"].append(loss)
            zs["Loss (initial)"].append(init_loss)

        if skip_flag:
            logger.warning(f"Skipping {figure_name} because not enough data")
            continue

        plot_order = [
            "NegativeSign (optimized)",
            "Loss (optimized)",
            "NegativeSign (initial)",
            "Loss (initial)",
        ]

        max_loss = max(np.max(zs["Loss (optimized)"]), np.max(zs["Loss (initial)"]))
        if np.abs(max_loss) < 1e-5:
            max_loss = 1

        fig, ax = plt.subplots(2, 2, figsize=(8, 8))
        for i, (key, z) in enumerate(zs.items()):
            Z = np.array(z).reshape(x.shape)
            # Determine the vmin and vmax based on whether 'Loss' is in the key
            vmin, vmax = (0, max_loss) if "Loss" in key else (0, 1)

            # Use imshow with an appropriate aspect ratio to ensure square cells
            c = ax[i % 2, i // 2].imshow(Z, cmap='RdPu', vmin=vmin, vmax=vmax, aspect='1',
                                         extent=[x.min(), x.max(), y.min(), y.max()],
                                         origin='lower', interpolation='none')

            ax[i % 2, i // 2].set_title(key)
            ax[i % 2, i // 2].set_xlabel('Jx/Jz')
            ax[i % 2, i // 2].set_ylabel('Jy/Jz')
            
            fig.colorbar(c, ax=ax[i % 2, i // 2])

        plt.tight_layout()  # Adjust subplots to fit into the figure area.
        figure_path.parent.mkdir(parents=False, exist_ok=True)
        plt.savefig(figure_path, bbox_inches='tight')
        logger.info(f"Figure saved to {figure_path}")

elif model_name == "MG1D":

    df = df0
    T_list = np.sort(df.temp.unique())
    L_list = np.sort(df.n_sites.unique())

    for T, L in itertools.product(T_list, L_list):

        # Determine a figure name that reflects the model and parameters
        figure_name = f"AsHeatmap_T_{T}_L_{L}.png"
        figure_path = image_model_dir / figure_name
        logger.info(f"Plotting {figure_name}")

        df = df0[(df0.temp == T) & (df0.n_sites == L)]

        J1_list = np.sort(df.J1.unique())
        J2_list = np.sort(df.J2.unique())
        J3_list = np.sort(df.J3.unique())

        zs = {
            "NegativeSign (optimized)": [],
            "NegativeSign (initial)": [],
            "Loss (optimized)": [],
            "Loss (initial)": [],
        }
        x, y = np.meshgrid(J2_list, J3_list)
        skip_flag = False
        for J2, J3 in zip(x.reshape(-1), y.reshape(-1)):
            df_plot = df
            df_plot = df_plot[(df_plot.J2 == J2) & (df_plot.J3 == J3)]
            df_u = df_plot[~df_plot.loss.isna()]
            df_h = df_plot[df_plot.loss.isna()]
            if len(df_u) == 0 or len(df_h) == 0:
                logger.debug(f"Skipping J2={J2}, J3={J3}")
                skip_flag = True
                break
        #     au = df_u["as"].min()
            ah = df_h["as"].min()
            ah_err = df_h["as_error"].min() * np.sqrt(N)
            idx = np.argmin(df_u.loss.values)
            loss = df_u.loss.values[idx]
            init_loss = df_u.init_loss.values[idx]
            au = df_u["as"].values[idx]
            au_err = df_u["as_error"].values[idx] * np.sqrt(N)

            zs["NegativeSign (optimized)"].append(au_err/au)
            zs["NegativeSign (initial)"].append(ah_err/ah)
            zs["Loss (optimized)"].append(loss)
            zs["Loss (initial)"].append(init_loss)

        if skip_flag:
            logger.warning(f"Skipping {figure_name} because not enough data")
            continue

        plot_order = [
            "NegativeSign (optimized)",
            "Loss (optimized)",
            "NegativeSign (initial)",
            "Loss (initial)",
        ]

        max_loss = max(np.max(zs["Loss (optimized)"]), np.max(zs["Loss (initial)"]))
        max_neg = max(np.max(zs["NegativeSign (optimized)"]), np.max(zs["NegativeSign (initial)"]))
        max_neg = min(max_neg, 100)
        if np.abs(max_loss) < 1e-5:
            max_loss = 1

        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        for i, (key, z) in enumerate(zs.items()):
            Z = np.array(z).reshape(x.shape)
            # Determine the vmin and vmax based on whether 'Loss' is in the key
            vmin, vmax = (0, max_loss) if "Loss" in key else (0, max_neg)

            # Use imshow with an appropriate aspect ratio to ensure square cells
            c = ax[i % 2, i // 2].imshow(Z, cmap='RdPu', vmin=vmin, vmax=vmax, aspect='1',
                                         extent=[x.min(), x.max(), y.min(), y.max()],
                                         origin='lower', interpolation='none')

            ax[i % 2, i // 2].set_title(key)
            ax[i % 2, i // 2].set_xlabel('J2/J1')
            ax[i % 2, i // 2].set_ylabel('J3/J1')
            fig.colorbar(c, ax=ax[i % 2, i // 2])

        plt.tight_layout()  # Adjust subplots to fit into the figure area.
        figure_path.parent.mkdir(parents=False, exist_ok=True)
        plt.savefig(figure_path, bbox_inches='tight')
        logger.info(f"Figure saved to {figure_path}")
