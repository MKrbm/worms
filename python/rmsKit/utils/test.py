from path import extract_info_from_txt, find_summary_files
from pathlib import Path
import pandas as pd
import logging

path = Path("array/quetta/HXYZ1D_loc")

summary_files = find_summary_files(path)
N = 10**6


def get_df_from_summary_files(summary_files):
    df = None
    for file in summary_files:
        if "sweeps_{}".format(N) in str(file):
            if df is None:
                df = pd.read_csv(file)
            else:
                try:
                    df = pd.concat([df, pd.read_csv(file)])
                except BaseException as e:
                    logging.warning("Could not concate or read {}".format(file))
                    logging.warning(e)
    return df


df = get_df_from_summary_files(summary_files)
df = df[df.n_sites == N]
