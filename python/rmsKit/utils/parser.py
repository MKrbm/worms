import argparse
import random
import logging

logger = logging.getLogger(__name__)

models = [
    "KH2D",
    "HXYZ1D",
    "HXYZ2D",
    "FF1D",
    "FF2D",
    "BLBQ1D",
    "MG1D",
    "SS2D",
]
# minimum energy solver, quasi energy solver
loss_val = ["mel", "none", "qsmel", "stoq"]


def get_parser(length: bool = False, model=None, Description=None):

    parser = argparse.ArgumentParser(
        description=Description)
    if model is None:
        parser.add_argument("-m", "--model", help="model (model) Name",
                            required=True, choices=models)
    else:
        pass
    parser.add_argument("-Jz", "--coupling_z", help="coupling constant (Jz)",
                        type=float, default=1)  # SxSx + SySy +
    parser.add_argument("-Jx", "--coupling_x",
                        help="coupling constant (Jx)", type=float)
    parser.add_argument("-Jy", "--coupling_y",
                        help="coupling constant (Jy)", type=float)
    parser.add_argument("-hx", "--mag_x", help="magnetic field",
                        type=float, default=0)
    parser.add_argument("-hz", "--mag_z", help="magnetic field",
                        type=float, default=0)
    parser.add_argument("-T", "--temperature", help="temperature", type=float)
    parser.add_argument("-M", "--num_iter",
                        help="# of iterations", type=int, default=10)
    parser.add_argument("-r", "--seed", help="random seed",
                        type=int, default=random.randint(0, 1000000))
    parser.add_argument("--sps", help="sps", type=int, default=3)
    parser.add_argument("-lr", "--learning_rate",
                        help="learning rate", type=float, default=0.01)
    parser.add_argument(
        "--stdout", help="print out the result in system terminal", action="store_true")

    parser.add_argument("--obc", help="open boundary condition", action="store_true")

    parser.add_argument("-e", "--epochs", help="epoch", type=int, default=100)
    parser.add_argument(
        "-lt",
        "--lattice_type",
        help="algorithm determine local unitary matrix",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-loss",
        "--loss",
        help="loss_methods",
        choices=loss_val,
        default="mel",
        nargs="?",
        const="all",
    )

    parser.add_argument(
        "--dtype",
        help="dtype of the unitary matrix. Default is float64",
        choices=["float64", "complex128", "float32", "complex64"],
        default="float64",
        nargs="?",
    )

    parser.add_argument(
        "-o",
        "--optimizer",
        help="optimizer",
        choices=["LION", "Adam"],
        default="Adam",
        nargs="?",
        const="all",
    )

    parser.add_argument(
        "-p",
        "--platform",
        help="cpu / gpu",
        choices=["cpu", "gpu"],
        nargs="?",
        const="all",
        default="cpu",
    )

    parser.add_argument(
        "-n",
        "--num_threads",
        help="number of threads",
        type=int,
        default=1,
    )

    parser.add_argument(
        "-J1",
        "--J1",
        help="first neighbor coupling",
        type=float,
        default=1,
    )

    parser.add_argument(
        "-J2",
        "--J2",
        help="second neighbor coupling",
        type=float,
        default=1,
    )

    parser.add_argument(
        "-J3",
        "--J3",
        help="third neighbor coupling",
        type=float,
        default=1,
    )

    parser.add_argument(
        "-J0",
        "--J0",
        help="neighbor coupling",
        type=float,
        default=1,
    )

    parser.add_argument('-L1', "--length1", help="length of side", type=int, required=length)
    parser.add_argument('-L2', "--length2", help="length of side", type=int)

    return parser


def get_params_parser(parser):

    args = parser.parse_args()

    params = dict(
        J0=args.J0,
        J1=args.J1,
        J2=args.J2,
        J3=args.J3,
        Jx=args.coupling_x if args.coupling_x is not None else args.coupling_z,
        Jy=args.coupling_y if args.coupling_y is not None else args.coupling_z,
        Jz=args.coupling_z,
        hx=args.mag_x,
        hz=args.mag_z,
        lt=args.lattice_type,
        sps=args.sps,
        seed=args.seed,  # random seed to generate hamiltonian
        obc=args.obc,
    )

    args_str = "args: {}".format(args)
    hash_str = str(hash(args_str))

    return args, params, hash_str

def get_dtype(dtype):
    import torch

    if dtype == "float64":
        return torch.float64
    elif dtype == "complex128":
        return torch.complex128
    elif dtype == "float32":
        return torch.float32
    elif dtype == "complex64":
        return torch.complex64
    else:
        raise TypeError("dtype not supported")

def print_args_info(args):
    """
    Print out the argument information.
    """
    logger.info("Argument Information:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")

