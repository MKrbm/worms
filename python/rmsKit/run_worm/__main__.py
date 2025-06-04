import sys
import numpy as np
import os
import logging
import re
import pandas as pd
from pathlib import Path
import subprocess

# ensure rmsKit is on PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils  # noqa: E402
from utils.parser import get_parser  # noqa: E402
from utils import get_logger, extract_info_from_file  # noqa: E402

# --------------------------- Argument Parser ---------------------------
parser = get_parser()
logger = get_logger("log.log", stdout=True, level=logging.INFO)

parser.add_argument(
    '-f', '--path', type=str, required=True,
    help='Directory containing Hamiltonian and best unitary files.'
)
parser.add_argument(
    '-s', '--sweeps', type=int, required=True,
    help='Number of sweeps to perform.'
)
parser.add_argument(
    '--original', action='store_true', default=False,
    help='Use the original Hamiltonian (no learned unitary).'
)
parser.add_argument(
    '-k', '--top_k', type=int, default=1,
    help='Number of top unitary paths to consider.'
)
parser.add_argument(
    '--system_size', action='store_true', default=False,
    help='Loop over system sizes in the outer loop.'
)
parser.add_argument(
    '--detail_points', type=int, default=0,
    help='Number of intermediate points to simulate on high-negativity; 0 to disable.'
)

args = parser.parse_args()
search_path = Path(args.path)

# --------------------------- Initial Checks ---------------------------
if not search_path.is_dir():
    logger.error("Given search path does not exist: %s", args.path)
    sys.exit(1)

if search_path.is_symlink():
    resolved = search_path.resolve()
    logger.warning("Resolving symbolic link %s -> %s", search_path, resolved)
    search_path = resolved

rmsKit_directory = Path(__file__).resolve().parent.parent

# --------------------------- Model Configuration ---------------------------
if __name__ == "__main__":
    beta_select = None
    L_list_select = None
    if args.model == "SS2D":
        if args.system_size:
            sizes = np.round(1.3 ** np.arange(3, 13)).astype(int)
            L_list = [[l, l] for l in sizes]
            beta = np.array([1])
        else:
            beta = np.array([0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            L_list = [[4, 4]]
        beta_select = 1
        L_list_select = [2, 2]
        logger.info("RUN SS2D MODEL")
    elif args.model == "HXYZ2D":
        beta = np.array([0.5, 1, 4])
        L_list = [[3, 3], [4, 4]]
        logger.info("RUN HXYZ2D MODEL")
    elif args.model == "KH2D":
        beta = np.array([1, 4])
        L_list = [[4, 4], [5, 5]]
        logger.info("RUN KH2D MODEL")
    elif args.model == "BLBQ1D":
        if args.system_size:
            sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
            L_list = [[l] for l in sizes]
            beta = np.array([1])
        else:
            beta = np.array([0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            L_list = [[10]]
        beta_select = 1
        L_list_select = [10]
        logger.info("RUN BLBQ1D MODEL")
    elif args.model == "FF2D":
        beta = np.array([1])
        L_list = [[4, 4]]
        logger.info("RUN FF2D MODEL")
    elif args.model == "MG1D":
        if args.system_size:
            sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
            L_list = [[l] for l in sizes]
            beta = np.array([1])
        else:
            beta = np.array([0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            L_list = [[10]]
        beta_select = 1
        L_list_select = [10]
        logger.info("RUN MG1D MODEL")
    else:
        raise ValueError(f"Model {args.model} is not supported")

    T_list = 1.0 / beta
    p = args.num_threads
    M = args.sweeps

    if M % p != 0:
        logger.info("Warning: M is not divisible by p, rounding down.")
        M = (M // p) * p

    min_loss, init_loss, ham_path, info_txt_path = utils.path.get_worm_path(search_path)
    simu_setting = f"sweeps_{M}_p_{p}" + ("_original" if args.original else "")
    summary_dir = info_txt_path.parent / "summary"
    save_dir = summary_dir / simu_setting
    save_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    while (save_dir / f"{idx}.csv").exists(): idx += 1
    save_path = save_dir / f"{idx}.csv"

    logger.info(f"min_loss: {min_loss}")
    logger.info(f"ham_path: {ham_path}")
    logger.info(f"L_list: {L_list}")
    logger.info(f"T_list: {T_list}")
    logger.info(f"Summary output: {save_path}")

    # --------------------------- Top-K Unitary Selection ---------------------------
    if args.top_k > 1 and beta_select is not None and not args.original:
        logger.info(f"Selecting top {args.top_k} unitary paths.")
        top_k_unitary_paths = utils.path.top_k_upath(search_path, args.top_k)
        neg_vals = []
        for loss, path in top_k_unitary_paths:
            logger.info(f"Testing path (loss={loss}): {path}")
            proc = utils.run_worm(
                args.model, ham_path, path, L_list_select,
                1.0 / beta_select, 10**4,
                n=p, logging=True, obc=args.obc,
                project_dir=rmsKit_directory.parent.parent.resolve()
            )
            output = proc.stdout.decode('utf-8')
            match = re.search(r'The result will be written in : "(.+?\.txt)"', output)
            try:
                data = extract_info_from_file(match.group(1), warning=True, allow_missing=False)
                neg_vals.append((data['as'], loss, path))
                logger.info(f"Avg sign = {data['as']}")
            except Exception as e:
                logger.error(f"Unitary path test failed: {e}")
        neg_vals.sort(key=lambda x: -x[0])
        min_path = neg_vals[0][2]
        logger.info(f"Selected unitary path: {min_path}")
    else:
        if args.original:
            min_path = ''
            logger.info("Using original Hamiltonian only.")
        else:
            min_path = None
            logger.info("Using lowest-loss Hamiltonian without unitary.")

    # --------------------------- Main Simulation Loop ---------------------------
    data_list = []
    outer_vals = T_list if args.system_size else L_list
    inner_vals = L_list if args.system_size else T_list
    outer_name, inner_name = ('T','L') if args.system_size else ('L','T')
    L_n_rel, T_n_rel = np.inf, 0

    for outer in outer_vals:
        logger.info(f"Starting sweep on {outer_name}={outer}")
        last_successful = [2] * len(inner_vals) if args.system_size else 100
        for inner in inner_vals:
            logger.info(f"Running simulation at {inner_name}={inner}")
            L = inner if args.system_size else outer
            T = outer if args.system_size else inner

            if L_n_rel <= (L[0] if isinstance(L,list) else L) and T_n_rel >= T:
                logger.info(f"Skipping unreliable region at L={L}, T={T}")
                continue

            proc = utils.run_worm(
                args.model, ham_path,
                '' if args.original else min_path,
                L, T, M, n=p, logging=True,
                obc=args.obc,
                project_dir=rmsKit_directory.parent.parent.resolve()
            )
            out = proc.stdout.decode('utf-8')
            match = re.search(r'The result will be written in : "(.+?\.txt)"', out)
            try:
                result_file = match.group(1)
                logger.info(f"Result file saved to: {result_file}")
                data = extract_info_from_file(result_file, warning=True, allow_missing=False)
                data['loss_func'] = info_txt_path.parent.name
                data_list.append(data)
                ratio = data['as_error']/data['as'] if data['as']!=0 else np.inf
                logger.info(f"Result: as={data['as']}, as_error={data['as_error']}, ratio={ratio:.3f}")

                if ratio>0.12 or data['as']<0:
                    logger.warning(f"High negativity detected at {inner_name}={inner}, ratio={ratio:.3f}")
                    if args.detail_points>0 and last_successful is not None:
                        v0 = (last_successful[0] if isinstance(last_successful,list)
                              else last_successful)
                        v1 = (inner[0] if isinstance(inner,list) else inner)
                        mids = np.linspace(v0, v1, args.detail_points+2)[1:-1]
                        break_outer=False
                        for mid in mids:
                            logger.info(f"Detail sampling at {inner_name}={mid}")
                            new_inner = [int(round(mid))] if isinstance(inner,list) else mid
                            Lm = new_inner if args.system_size else outer
                            Tm = outer if args.system_size else new_inner
                            proc_m = utils.run_worm(
                                args.model, ham_path,
                                '' if args.original else min_path,
                                Lm, Tm, M, n=p, logging=True,
                                obc=args.obc,
                                project_dir=rmsKit_directory.parent.parent.resolve()
                            )
                            out_m = proc_m.stdout.decode('utf-8')
                            match_m = re.search(r'The result will be written in : "(.+?\.txt)"', out_m)
                            try:
                                mid_file = match_m.group(1)
                                logger.info(f"Detail result file: {mid_file}")
                                dm = extract_info_from_file(mid_file, warning=True, allow_missing=False)
                                dm['loss_func'] = info_txt_path.parent.name
                                data_list.append(dm)
                                rmid = dm['as_error']/dm['as'] if dm['as']!=0 else np.inf
                                logger.info(f"Mid-result ratio={rmid:.3f}")
                                if rmid>0.12 or dm['as']<0:
                                    logger.warning(f"Break mid sampling at {inner_name}={mid}")
                                    break_outer=True
                                    L_n_rel = (Lm[0] if isinstance(Lm,list) else Lm)
                                    T_n_rel = Tm
                                    break
                            except Exception as e:
                                logger.error(f"Detail point failed at {mid}: {e}")
                        if break_outer:
                            break
                    L_n_rel = (L[0] if isinstance(L,list) else L)
                    T_n_rel = T
                    logger.info(f"Marking cutoff L_n_rel={L_n_rel}, T_n_rel={T_n_rel}")
                    break
                else:
                    last_successful = inner
                    logger.info(f"Successful run at {inner_name}={inner}")
            except Exception as e:
                logger.error(f"Simulation error at L={L}, T={T}: {e}")
                continue

    # --------------------------- Save Summary ---------------------------
    df = pd.DataFrame(data_list)
    df.to_csv(save_path, index=False)
    logger.info(f"Summary saved to : {save_path}")
