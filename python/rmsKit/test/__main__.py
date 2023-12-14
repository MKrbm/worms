import logging
import os
import pathlib
from textwrap import dedent
import subprocess
import contextlib
from typing import List, Any, Tuple
import numpy as np
import pandas as pd

from .runner.HXYZ import run_HXYZ1D
from .runner.BLBQ import run_BLBQ1D


def test_sim_res(df_u, df_h, df_e, df_s, eigs):

    test_fail = True

    diff = np.abs(np.sort(eigs) - np.sort(df_e.value)).mean()
    if not diff < 1e-8:
        logging.warning("HPhi test failed")
        logging.warning("diff: {}".format(diff))
        test_fail = False

    # test solver
    return test_solver_worm(df_u, df_h, df_e, df_s) and test_fail


def test_solver_worm(df_u, df_h, df_e, df_s):

    test_fail = True

    if not np.all(df_u.c_test) & np.all(df_u.e_test):
        logging.warning("optimized unitary worm failed solver test")
        logging.warning("C: \n{}".format(df_u[["specific_heat", "c", "c_error", "c_test"]]))
        logging.warning("E: \n{}".format(df_u[["energy_per_site", "e", "e_error", "e_test"]]))
        test_fail = False

    if not np.all(df_h.c_test) & np.all(df_h.e_test):
        logging.warning("identity worm failed solver test")
        logging.warning("C: \n{}".format(df_h[["specific_heat", "c", "c_error", "c_test"]]))
        logging.warning("E: \n{}".format(df_h[["energy_per_site", "e", "e_error", "e_test"]]))
        test_fail = False

    return test_fail


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
    output_dir = output_dir.as_posix()
    logging.debug("output_dir: {}".format(output_dir))
    return run_HXYZ1D(params, rmsKit_directory, output_dir)


def _run_BLBQ1D(Js: List[float],
                Hs: List[float],
                L: int,
                lt: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # run HXYZ1D
    params = {
        "L1": L,
        "J0": Js[0],
        "J1": Js[1],
        "hz": Hs[0],
        "hx": Hs[1],
        "lt": lt,
    }
    rmsKit_directory = pathlib.Path(
        __file__).resolve().parent.parent.as_posix()
    output_dir = FILE_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    output_dir = output_dir.as_posix()
    logging.debug("output_dir: {}".format(output_dir))
    return run_BLBQ1D(params, rmsKit_directory, output_dir)


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

    '''
    HPhi_in = """
        L={}
        model = "Spin"
        method = "FullDiag"
        lattice = "chain"
        Jx = {}
        Jy = {}
        Jz = {}
        H = {}
        2Sz = {{}} """.format(L, Js[0], Js[1], Js[2], H)
    HPhi_in = dedent(HPhi_in)
    eigs = np.sort(call_HPhi(HPhi_in, L))
    '''

    mdfu, mdfh, dfe, dfs = _run_HXYZ1D(Js, Hs, L)

    # f: test_sim_res(mdfu, mdfh, dfe, dfs, eigs)
    if test_solver_worm(mdfu, mdfh, dfe, dfs):
        logging.info("HXYZ1D(J=1.0) test passed")

    H = 0.3
    Hs = [H, 0]
    logging.info("Js: {} / Hs: {}".format(Js, Hs))
    mdfu, mdfh, dfe, dfs = _run_HXYZ1D(Js, Hs, L)
    if test_solver_worm(mdfu, mdfh, dfe, dfs):
        logging.info("HXYZ1D(hz = 0.3) test passed")

    L = 9
    Js = [-0.3, 0.5, 0.8]
    Hs = [0.3, 0]
    logging.info("Js: {}  Hs: {} / L = {}".format(Js, Hs, L))
    mdfu, mdfh, dfe, dfs = _run_HXYZ1D(Js, Hs, L)
    if test_solver_worm(mdfu, mdfh, dfe, dfs):
        logging.info("HXYZ1D(-Jx -0.3 -Jy 0.8 -Jz 0.5 -hx 0.3 -hz 0) test passed")

    # n: get the energy at beta = 1
    e = dfs.loc[dfs["beta"] == 1, "energy_per_site"].values[0]
    if not np.abs(e - -0.07947479512910453) < 1e-8:
        logging.warning("energy at beta = 1 is incorrect: {} != -0.07947479512910453".format(e))

    # n: run BLBQ1D test

    L = 6

    # n: AKLT model
    Js = [1, 1/3]
    Hs = [0, 0]
    logging.info("Run BLBQ1D test")

    mdfu, mdfh, dfe, dfs = _run_BLBQ1D(Js, Hs, L, 1)
    if test_solver_worm(mdfu, mdfh, dfe, dfs):
        logging.info("BLBQ1D(AKLT) test passed")

    e0 = np.min(dfe.value)
    analytic_e0 = -  L * (2/3)
    if np.abs(e0 - analytic_e0) < 1e-8:
        logging.info("ground state energy at AKLT point is correct : {} = - L * 2/3".format(e0))
    else:
        logging.warning(
            "ground state energy at AKLT point is incorrect : {} != - L * 2/3".format(e0))

    # n: alpha = 1
    Js = [1, 1]
    Hs = [0, 0]

    mdfu, mdfh, dfe, dfs = _run_BLBQ1D(Js, Hs, L, 1)
    if test_solver_worm(mdfu, mdfh, dfe, dfs):
        logging.info("BLBQ1D(alpha=1) test passed")

    # n: 33
    L = 3
    mdfu2, mdfh2, dfe2, dfs2 = _run_BLBQ1D(Js, Hs, L, 2)
    if np.linalg.norm(np.sort(dfe2.value) - np.sort(dfe.value)) < 1e-8:
        logging.info("ground state energy at lt=1 and L = 6 is consistent with lt=2 and L=3")
    else:
        logging.warning(
            "It is expected that the ground state energy at lt=1 and L = 6 is different from lt=2 and L=3")
        logging.warning("\n{} \n{}".format(dfe2.value,  dfe.value))

    if test_solver_worm(mdfu2, mdfh2, dfe2, dfs2):
        logging.info("BLBQ1D(alpha=1 / lt = 2) test passed")
