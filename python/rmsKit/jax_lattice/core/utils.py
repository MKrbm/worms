import numpy as np
import os
def proj_symm(x):
    s = int(np.sqrt(x.shape[0]))
    x = x.reshape(s,s,s,s)
    return ((x + np.einsum("ijkl->jilk", x))/2).reshape(s*s, s*s)


def save_npy(folder, hams):
    if isinstance(hams, list):
        if not os.path.exists(folder):
            os.makedirs(folder)
        for i, ham in enumerate(hams):
            name = f"{i}"
            np.save(folder + f"/{i}", (ham.real).astype(np.float64))
            print(f"save matrix ({ham.shape}): ", folder+ "/" + name + ".npy")
    else:
        if not os.path.exists(os.path.dirname(folder)):
            os.makedirs(os.path.dirname(folder))
        name = "0"
        np.save(folder, hams.real.astype(np.float64))
        print(f"save matrix ({hams.shape}): ", folder+ ".npy")