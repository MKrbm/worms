import numpy as np
def exp_energy(E, beta):
    Z = np.exp(-beta*E)
    EZ = E*Z
    return np.sum(EZ)/np.sum(Z)