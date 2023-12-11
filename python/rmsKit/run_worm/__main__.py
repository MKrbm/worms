import sys
import numpy as np
import os
import logging
import re
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils  # noqa: E402
from utils.parser import get_parser  # noqa: E402
from utils import get_logger, extract_info_from_file  # noqa: E402


parser = get_parser()
logger = get_logger("log.log", stdout=True, level=logging.DEBUG)

parser.add_argument(
    '-f',
    '--path',
    type=str,
    required=True,
    help='The python file will look for the Hamiltonian and the best unitary in this directory.')
parser.add_argument('-s', '--sweeps', type=int, required=True,
                    help='The number of sweeps to perform.')
parser.add_argument('--original', action='store_true', default=False,
                    help='If this flag is set, the original Hamiltonian will be used.')
args = parser.parse_args()
search_path = Path(args.path)

# check if given path exists
if not os.path.isdir(args.path):
    logging.debug("current dir is: ", os.getcwd())
    logging.info("given serch path doesn't exit.")
    sys.exit()

rmsKit_directory = Path(os.path.abspath(__file__)).parent.parent


if __name__ == "__main__":

    # define parameters list to be passed to the run_worm function
    beta = np.linspace(0.5, 2, 31)
    # beta = np.array([1, 2, 4])
    T_list = 1/beta

    # define the lattice sizes
    # L_list = [[i, i] for i in range(4, 11)]
    L_list = [[i] for i in range(11, 110, 10)]
    L_list += [[i+1] for i in range(10, 110, 10)]

    # define the number of samples
    p = args.num_threads
    M = args.sweeps

    if M % p != 0:
        logger.info(
            "Warning: M is not divisible by p. The number of sweeps will be rounded down.")
        M = (M // p) * p

    min_path, min_loss, ham_path = utils.path_with_lowest_loss(
        args.path, return_ham=True, absolute_path=True)
    ("ham_path: ", ham_path)
    # logger.info("min_path: ", min_path)
    # logger.info("min_loss: ", min_loss)
    # logger.info("ham_path: ", ham_path)
    logger.info("min_path: {}".format(min_path))
    logger.info("min_loss: {}".format(min_loss))
    logger.info("ham_path: {}".format(ham_path))

    logger.info("model_name: {}".format(args.model))
    logger.info(
        "Search path : {}".format(
            search_path.resolve().as_posix()))
    logger.info("L_list: {}".format(L_list))
    logger.info("T_list: {}".format(T_list))

    # run the simulation
    for L in L_list:
        for T in T_list:
            subprocess_out = utils.run_worm(
                args.model,
                ham_path,
                min_path if not args.original else "",
                L,
                T,
                M,
                n=p,
                logging=True,
                project_dir=rmsKit_directory.parent.parent.resolve())
            output = subprocess_out.stdout.decode("utf-8")
            # Extract the path using regex
            match = re.search(r'The result will be written in : "(.+?\.txt)"', output)
            try:
                result_file_path = match.group(1)
                # with open(result_file_path, 'r') as file:
                #     file_content = file.read()
                extracted_info = extract_info_from_file(
                    result_file_path, warning=True, allow_missing=False)
                print(extracted_info)
            except Exception as e:
                logger.error("Exception: {}".format(e))
                logger.error(
                    "No result file found. This may be due to an error in the logging in main_MPI.")
                logger.error("subprocess_out: {}".format(output))
                continue