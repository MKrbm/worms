import numpy as np
def proj_symm(x):
    s = int(np.sqrt(x.shape[0]))
    x = x.reshape(s,s,s,s)
    return ((x + np.einsum("ijkl->jilk", x))/2).reshape(s*s, s*s)

import os

def save_npy(folder, hams):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i, ham in enumerate(hams):
        name = f"{i}"
        np.save(folder + f"/{i}", (ham.real).astype(np.float64))
        print(f"save matrix ({ham.shape}): ", folder+ "/" + name + ".npy")