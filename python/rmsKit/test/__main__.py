from utils.parser import models as MODELS
from .runner.BLBQ import run_BLBQ1D
from .runner.HXYZ import run_HXYZ
from .runner.MG import run_MG1D
from .runner.SS import run_SS2D
from .runner.KH import run_KH2D
import logging
import os
from pathlib import Path
from textwrap import dedent
import subprocess
import contextlib
from typing import List, Any, Tuple
import numpy as np
import pandas as pd
import sys
import argparse
sys.path.append(str(Path(__file__).resolve().parent.parent))
parser = argparse.ArgumentParser(
    description="Test exact diagonalization solver, worm algorithm by comparing with analytical results")
MODELS.append("all")
parser.add_argument("-m", "--model", help="model (model) Name",
                    required=True, choices=MODELS)
model = parser.parse_args().model


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
        logging.warning("C: \n{}".format(df_u[["beta", "specific_heat", "c", "c_error", "c_test"]]))
        logging.warning("E: \n{}".format(
            df_u[["beta", "energy_per_site", "e", "e_error", "e_test"]]))
        test_fail = False
    else:
        logging.info("E: \n{}".format(df_u[["beta", "energy_per_site", "e", "e_error", "e_test"]]))

    if not np.all(df_h.c_test) & np.all(df_h.e_test):
        logging.warning("identity worm failed solver test")
        logging.warning("C: \n{}".format(df_h[["beta", "specific_heat", "c", "c_error", "c_test"]]))
        logging.warning("E: \n{}".format(
            df_h[["beta", "energy_per_site", "e", "e_error", "e_test"]]))
        test_fail = False
    else:
        logging.info("E: \n{}".format(df_h[["beta", "energy_per_site", "e", "e_error", "e_test"]]))

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
    rmsKit_directory = Path(
        __file__).resolve().parent.parent.as_posix()
    output_dir = FILE_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    output_dir = output_dir.as_posix()
    logging.debug("output_dir: {}".format(output_dir))
    return run_HXYZ(params, rmsKit_directory, output_dir)


def _run_HXYZ2D(Js: List[float],
                Hs: List[float],
                L1: int, L2: int) -> Tuple[pd.DataFrame,
                                           pd.DataFrame,
                                           pd.DataFrame,
                                           pd.DataFrame]:
    # run HXYZ1D
    params = {
        "L1": L1,
        "L2": L2,
        "Jz": Js[0],
        "Jx": Js[1],
        "Jy": Js[2],
        "hz": Hs[0],
        "hx": Hs[1],
    }

    rmsKit_directory = Path(
        __file__).resolve().parent.parent.as_posix()
    output_dir = FILE_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    output_dir = output_dir.as_posix()
    logging.debug("output_dir: {}".format(output_dir))
    return run_HXYZ(params, rmsKit_directory, output_dir)


def _run_BLBQ1D(Js: List[float],
                Hs: List[float],
                L: int,
                lt: int,
                obc: bool = False,
                ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # run HXYZ1D
    params = {
        "L1": L,
        "J0": Js[0],
        "J1": Js[1],
        "hz": Hs[0],
        "hx": Hs[1],
        "lt": lt,
        "obc": obc,
    }
    rmsKit_directory = Path(
        __file__).resolve().parent.parent.as_posix()
    output_dir = FILE_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    output_dir = output_dir.as_posix()
    logging.debug("output_dir: {}".format(output_dir))
    return run_BLBQ1D(params, rmsKit_directory, output_dir)


def _run_MG1D(Js: List[float],
              L: int,
              lt: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # run HXYZ1D
    params = {
        "L1": L,
        "J1": Js[0],
        "J2": Js[1],
        "J3": Js[2],
        "lt": lt,
    }
    rmsKit_directory = Path(
        __file__).resolve().parent.parent.as_posix()
    output_dir = FILE_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    logging.debug("output_dir: {}".format(output_dir.as_posix()))
    return run_MG1D(params, rmsKit_directory, output_dir.as_posix())


def _run_SS2D(Js: List[float],
              L1: int, L2: int) -> Tuple[pd.DataFrame,
                                         pd.DataFrame,
                                         pd.DataFrame,
                                         pd.DataFrame]:
    # run HXYZ1D
    params = {
        "L1": L1,
        "L2": L2,
        "J0": Js[0],
        "J1": Js[1],
        "J2": Js[2],
        "lt": 1,
    }

    rmsKit_directory = Path(
        __file__).resolve().parent.parent.as_posix()
    output_dir = FILE_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    logging.debug("output_dir: {}".format(output_dir.as_posix()))
    return run_SS2D(params, rmsKit_directory, output_dir.as_posix())


def _run_KH2D(Js: List[float], hs: List[float],
              L1: int, L2: int) -> Tuple[pd.DataFrame,
                                         pd.DataFrame,
                                         pd.DataFrame,
                                         pd.DataFrame]:
    # run HXYZ1D
    params = {
        "L1": L1,
        "L2": L2,
        "Jz": Js[0],
        "Jx": Js[1],
        "Jy": Js[2],
        "hz": hs[0],
        "hx": hs[1],
        "lt": 3,
    }

    rmsKit_directory = Path(
        __file__).resolve().parent.parent.as_posix()
    output_dir = FILE_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    logging.debug("output_dir: {}".format(output_dir.as_posix()))
    return run_KH2D(params, rmsKit_directory, output_dir.as_posix())


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

FILE_DIR = Path(__file__).resolve().parent
HPHI_PATH = Path("HPhi")
STAT_IN = HPHI_PATH / "stat.in"
OUT_HPHI = HPHI_PATH / "output" / "Eigenvalue.dat"

CMD_HPHI = "HPhi -s stat.in"
CMD_TEST = "python {} -L1 {} -Jz {} -Jx {} -Jy {} -hx {} -hz {} --stdout"
ENV = os.environ.copy()
# change directory to FILE_DIR
os.chdir(FILE_DIR)
if __name__ == "__main__":

    # HXYZ1D
    if model == "HXYZ1D" or model == "all":
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
        if not np.abs(e - (-0.07947479512910453)) < 1e-8:
            logging.warning("energy at beta = 1 is incorrect: {} != -0.07947479512910453".format(e))

    # n: run HXYZ2D test
    if model == "HXYZ2D" or model == "all":

        logging.info("Run HXYZ1D2 test")
        Ls = [3, 3]
        Js = [1, 1, 1]
        Hs = [0, 0]
        logging.info("Js: {} / Hs: {}".format(Js, Hs))

        mdfu, mdfh, dfe, dfs = _run_HXYZ2D(Js, Hs, Ls[0], Ls[1])

        if test_solver_worm(mdfu, mdfh, dfe, dfs):
            logging.info("HXYZ1D2(J=1.0) test passed")

        Ls = [2, 4]
        Js = [-0.3, 0.5, 0.8]
        Hs = [0.3, 0]
        logging.info("Js: {}  Hs: {} / L = {}".format(Js, Hs, Ls))
        mdfu, mdfh, dfe, dfs = _run_HXYZ2D(Js, Hs, Ls[0], Ls[1])
        e = dfs.loc[dfs["beta"] == 1, "energy_per_site"].values[0]
        if not np.abs(e - (-0.18543629571195416)) < 1e-8:  # Check if exact solver is correct
            logging.warning("energy at beta = 1 is incorrect: {} != -0.07947479512910453".format(e))

        if test_solver_worm(mdfu, mdfh, dfe, dfs):
            logging.info("HXYZ1D2(-Jx -0.3 -Jy 0.8 -Jz 0.5 -hx 0.3 -hz 0) test passed")

        Ls = [3, 4]
        Js = [-0.7, 0.5, 0.8]
        Hs = [0.3, 0.2]
        logging.info("Js: {}  Hs: {} / Ls = {}".format(Js, Hs, Ls))
        mdfu, mdfh, dfe, dfs = _run_HXYZ2D(Js, Hs, Ls[0], Ls[1])
        if test_solver_worm(mdfu, mdfh, dfe, dfs):
            logging.info("HXYZ1D2(-Jx -0.7 -Jy 0.8 -Jz 0.5 -hx 0.3 -hz 0.2) test passed")

    # n: run MG1D test
    if model == "MG1D" or model == "all":

        L = 4

        # MG point
        Js = [0.5, 1, 1]
        logging.info("Run MG1D test")

        mdfu, mdfh, dfe, dfs = _run_MG1D(Js, L, 2)
        if test_solver_worm(mdfu, mdfh, dfe, dfs):
            logging.info("MG1D test1 passed")

        e0 = np.min(dfe.value)
        analytic_e0 = -  L * 2 * (3/4) * (1/2)
        if np.abs(e0 - analytic_e0) < 1e-8:
            logging.info("ground state energy at MG point is correct : {} = - L * 3 / 4".format(e0))
        else:
            logging.warning(
                "ground state energy at MG point is incorrect : {} != - L * 3 / 4".format(e0))

        Js = [1, 3/2, -1/2]

        logging.info("Js: {}".format(Js))

        mdfu, mdfh, dfe, dfs = _run_MG1D(Js, L, 2)
        if test_solver_worm(mdfu, mdfh, dfe, dfs):
            logging.info("MG1D test2 passed")

    # n: run BLBQ1D test
    if model == "BLBQ1D" or model == "all":

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

        # n: alpha = -0.3
        Js = [1, -0.3]
        Hs = [0, 0.3]
        L = 4

        mdfu, mdfh, dfe, dfs = _run_BLBQ1D(Js, Hs, L, 1, obc=True)
        if test_solver_worm(mdfu, mdfh, dfe, dfs):
            logging.info("BLBQ1D(alpha=1) test passed")

    if model == "SS2D" or model == "all":

        Ls = [1, 1]
        logging.info("Run SS2D test")

        # n: Extreme Dimer case
        Js = [1.0, 0., 0]
        logging.info("Js: {}".format(Js))
        mdfu, mdfh, dfe, dfs = _run_SS2D(Js,  Ls[0], Ls[1])
        if test_solver_worm(mdfu, mdfh, dfe, dfs):
            logging.info("SS2D(Disentangled, dimer basis is gs) test passed")

        e0 = np.min(dfe.value)
        analytic_e0 = -  Ls[0] * Ls[1] * (3/4) * 4
        if np.abs(e0 - analytic_e0) < 1e-8:
            logging.info(
                "ground state energy at Shastry-Sutherland point is correct : {} = - L * 3 / 4".format(e0))
        else:
            logging.warning(
                "ground state energy at Shastry-Sutherland point is incorrect : {} != - L * 3 / 4".format(e0))

        # n: Shastry-Sutherland model (singlet-product phase)
        Js = [1.0, 0.2, 0]
        logging.info("Js: {}".format(Js))
        mdfu, mdfh, dfe, dfs = _run_SS2D(Js,  Ls[0], Ls[1])
        if test_solver_worm(mdfu, mdfh, dfe, dfs):
            logging.info("SS2D(Singlet-product basis) test passed")

        e0 = np.min(dfe.value)
        N = 4 * Ls[0] * Ls[1]
        analytic_e0 = -  N * (3/4)
        if np.abs(e0 - analytic_e0) < 1e-8:
            logging.info(
                "ground state energy at Shastry-Sutherland point is correct : {} = - N * 3".format(e0))
        else:
            logging.warning(
                "ground state energy at Shastry-Sutherland point is incorrect : {} != - N * 3".format(e0))

        # n: Shastry-Sutherland model (also singlet-product phase)
        Js = [1.0, 0.2, 0.4]
        logging.info("Js: {}".format(Js))
        mdfu, mdfh, dfe, dfs = _run_SS2D(Js,  Ls[0], Ls[1])
        if test_solver_worm(mdfu, mdfh, dfe, dfs):
            logging.info("SS2D(Singlet-product basis) test passed")

        e0 = np.min(dfe.value)
        if np.abs(e0 - analytic_e0) < 1e-8:
            logging.info(
                "ground state energy at Shastry-Sutherland point is correct : {} = - N * 3".format(e0))
        else:
            logging.warning(
                "ground state energy at Shastry-Sutherland point is incorrect : {} != - N * 3".format(e0))

        # n: Shastry-Sutherland model (plaquette phase)
        Js = [1.0, 1, 0]
        logging.info("Run SS2D test")

        logging.info("Js: {}".format(Js))
        mdfu, mdfh, dfe, dfs = _run_SS2D(Js,  Ls[0], Ls[1])
        if test_solver_worm(mdfu, mdfh, dfe, dfs):
            logging.info("SS2D(yet dimer basis is gs) test passed")

    if model == "KH2D" or model == "all":

        Ls = [2, 2]
        logging.info("Run SS2D test")

        # n: Kagome Heisenberg model
        Js = [1.0, 1.0, 1.0]
        hs = [0.0, 0.0]
        logging.info("Js: {}".format(Js))
        mdfu, mdfh, dfe, dfs = _run_KH2D(Js, hs, Ls[0], Ls[1])
        if test_solver_worm(mdfu, mdfh, dfe, dfs):
            logging.info("KH2D test passed")


        # n: Kagome Heisenberg model
        Js = [1.0, - 0.8, 0.5]
        hs = [0.0, 0.3]
        logging.info("Js: {}".format(Js))
        mdfu, mdfh, dfe, dfs = _run_KH2D(Js, hs, Ls[0], Ls[1])
        if test_solver_worm(mdfu, mdfh, dfe, dfs):
            logging.info("KH2D test passed")
