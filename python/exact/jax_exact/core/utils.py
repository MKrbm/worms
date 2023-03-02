import numpy as np
def proj_symm(x):
    s = int(np.sqrt(x.shape[0]))
    x = x.reshape(s,s,s,s)
    return ((x + np.einsum("ijkl->jilk", x))/2).reshape(s*s, s*s)