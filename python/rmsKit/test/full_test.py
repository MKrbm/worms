import logging
import os
import pathlib
import subprocess
from textwrap import dedent
import contextlib
from typing import List, Any
import numpy as np


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


def run_HXYZ1D(Js: List[float], Hs: List[float], L: int) -> List[float]:
    cmd = CMD_TEST.format("HXYZ1D.py", L, *Js, *Hs)
    with change_directory(FILE_DIR):
        out = subprocess.run(
            cmd,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            shell=True,
            env=ENV)
    stdout = out.stdout.decode("utf-8")
    logging.info(stdout)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

FILE_DIR = pathlib.Path(__file__).resolve().parent.as_posix()
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

    J = 1.0
    H = 0.0
    Js = [J, J, J]
    Hs = [H, 0]
    logging.info("Js: {} / Hs: {}".format(Js, Hs))

    HPhi_in = """
        L={}
        model = "Spin"
        method = "FullDiag"
        lattice = "chain"
        J = {}
        H = {}
        2Sz = {{}} """.format(L, J, H)
    HPhi_in = dedent(HPhi_in)
    eigs = np.sort(call_HPhi(HPhi_in, L))
    
    run_HXYZ1D(Js, Hs, L)
