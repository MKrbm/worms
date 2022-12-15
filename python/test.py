# simulate quantum monte carlo for a 1D heisenberg chain with periodic boundary conditions

import numpy as np
import matplotlib.pyplot as plt
import time

# define the quantum hamiltonian
def hamiltonian(spin, J):
    H = 0
    for i in range(len(spin)-1):
        H += -J * spin[i] * spin[i+1]
    H += -J * spin[-1] * spin[0]
    return H
