import logging
import os
import pathlib
import subprocess
from textwrap import dedent
import contextlib
from typing import List, Any, Tuple
import numpy as np
import pandas as pd
from nptyping import NDArray

from .runner.HXYZ import run_HXYZ1D


def test_sim_res(df_u: pd.DataFrame,
                 df_h: pd.DataFrame,
                 df_e: pd.DataFrame,
                 df_s: pd.DataFrame,
                 eigs: NDArray = np.array([])):

    if len(eigs):
        diff = np.abs(np.sort(eigs) - np.sort(df_e.value)).mean()
        if diff < 1e-8:
            logging.info("HPhi test passed")
        else:
            logging.warning("HPhi test failed")
            logging.warning("diff: {}".format(diff))
    else:
        logging.warning("HPhi test skipped")

    if np.all(df_u.c_test) & np.all(df_u.e_test):
        logging.info("optimized unitary worm passed solver test")
    else:
        logging.warning("optimized unitary worm failed solver test")
        logging.warning("c_test: \n{}".format(df_u))
        logging.warning("e_test: \n{}".format(df_u))

    if np.all(df_h.c_test) & np.all(df_h.e_test):
        logging.info("identity worm passed solver test")
    else:
        logging.warning("identity worm failed solver test")
        logging.warning("c_test: \n{}".format(df_h))
        logging.warning("e_test: \n{}".format(df_h))


@contextlib.contextmanager
def change_directory(directory):
    original_directory = os.getcwd()
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(original_directory)


def call_HPhi(HPhi_in: str, L: int) -> List[float]:
    eigs = []
    for sz2 in range(0, L+1):
        # write the HPhi_in into a stat.in file
        with open(STAT_IN, "w") as f:
            f.write(HPhi_in.format(sz2))
        # print(HPhi_in.format(sz2))

        # run HPhi. Change directory to HPhi
        with change_directory(HPHI_PATH):
            out = subprocess.run(
                CMD_HPHI,
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE,
                shell=True,
                env=ENV)
            stdout = out.stdout.decode("utf-8")

        # load the output file
        eigs_tmp = []
        with open(OUT_HPHI, "r") as f:
            for line in f:
                # lines.append(line.strip())
                eigs_tmp.append(float(line.split()[1]))
        eigs += eigs_tmp
    return eigs


def _run_HXYZ1D(Js: List[float],
                Hs: List[float],
                L: int) -> Tuple[pd.DataFrame,
                                 pd.DataFrame,
                                 pd.DataFrame,
                                 pd.DataFrame]:
    # run HXYZ1D
    params = {
        "L1": L,
        "Jz": Js[0],
        "Jx": Js[1],
        "Jy": Js[2],
        "hz": Hs[0],
        "hx": Hs[1],
    }
    rmsKit_directory = pathlib.Path(
        __file__).resolve().parent.parent.as_posix()
    output_dir = FILE_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    logging.debug("output_dir: {}".format(output_dir.as_posix()))
    return run_HXYZ1D(params, rmsKit_directory, output_dir.as_posix())


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

FILE_DIR = pathlib.Path(__file__).resolve().parent
HPHI_PATH = pathlib.Path("HPhi")
STAT_IN = HPHI_PATH / "stat.in"
OUT_HPHI = HPHI_PATH / "output" / "Eigenvalue.dat"

CMD_HPHI = "HPhi -s stat.in"
CMD_TEST = "python {} -L1 {} -Jz {} -Jx {} -Jy {} -hx {} -hz {} --stdout"
ENV = os.environ.copy()
# change directory to FILE_DIR
os.chdir(FILE_DIR)
if __name__ == "__main__":

    # HXYZ1D
    L = 5
    logging.info("Run HXYZ1D test")
    logging.info("Compare solver_torch.py and HPhi")
    logging.info("Please make sure to install HPhi and set the environment variable HPHI_PATH before running this test.")  # noqa

    J = 1.0
    H = 0.0
    Js = [J, J, J]
    Hs = [H, 0]
    logging.info("Js: {} / Hs: {}".format(Js, Hs))

    HPHI_HXYZ = """
        L={}
        model = "Spin"
        method = "FullDiag"
        lattice = "chain"
        J = {}
        H = {}
        2Sz = {{}} """

    HPhi_in = dedent(HPHI_HXYZ.format(L, J, H))
    eigs = np.sort(call_HPhi(HPhi_in, L))
    mdfu, mdfh, dfe, dfs = _run_HXYZ1D(Js, Hs, L)
    test_sim_res(mdfu, mdfh, dfe, dfs, eigs)

    H = 0.3
    Hs = [H, 0]
    logging.info("Js: {} / Hs: {}".format(Js, Hs))
    HPhi_in = dedent(HPHI_HXYZ.format(L, J, H))
    eigs = np.sort(call_HPhi(HPhi_in, L))
    mdfu, mdfh, dfe, dfs = _run_HXYZ1D(Js, Hs, L)
    test_sim_res(mdfu, mdfh, dfe, dfs, eigs)

    Js = [-0.3, 0.5, 0.8]
    Hs = [0.3, 0]
    logging.info("Js: {} / Hs: {}".format(Js, Hs))
    mdfu, mdfh, dfe, dfs = _run_HXYZ1D(Js, Hs, L)
    test_sim_res(mdfu, mdfh, dfe, dfs)
