import sys
import numpy as np
import utils
import argparse
import os

# define the arguments
parser = argparse.ArgumentParser(description='Run a worm simulation.')

parser.add_argument('-m', '--model_name', type=str, required=True,
                    help='The name of the model.')
parser.add_argument('-f', '--path', type=str, required=True,
                    help='The python file will look for the Hamiltonian and the best unitary in this directory.')
parser.add_argument('-r', '--seed', type=int, required=True,
                    help='The seed for the random number generator.')
parser.add_argument('-n', '--n', type=int, default=1,
                    help='The number of processes to use.')
parser.add_argument('-s', '--sweeps', type=int, required=True,
                    help='The number of sweeps to perform.')

# parse the arguments
args = parser.parse_args()

# check if given path exists
if not os.path.isdir(args.path):
    print("current dir is: ", os.getcwd())
    print("path : ", args.path, " not found. Please check the path.")
    sys.exit()


if __name__ == "__main__":

    # define parameters list to be passed to the run_worm function
    beta = np.linspace(1, 5, 21)
    T_list = 1/beta

    # define the lattice sizes
    L_list = [[i, i] for i in range(4, 11)]

    # define the number of samples
    p = args.n
    M = args.sweeps
    r = args.seed

    if M % p != 0:
        print(
            "Warning: M is not divisible by p. The number of sweeps will be rounded down.")
        M = (M // p) * p

    path = args.path
    path = os.getcwd() + "/" + path


    min_path, min_loss, ham_path = utils.path_with_lowest_loss(
        args.path, return_ham=True, absolute_path=True)
    print("ham_path: ", ham_path)
    print("min_path: ", min_path)
    print("min_loss: ", min_loss)

    print("simulation parameters:")
    print("model_name: ", args.model_name)
    print("path: ", path)
    print("L_list: ", L_list)
    print("T_list: ", T_list)


    # run the simulation
    for L in L_list:
        for T in T_list:
            utils.run_worm(args.model_name, ham_path, min_path, L, T, M, n=p)
