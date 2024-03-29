import numpy as np
import os

def save_npy(folder, hams):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i, ham in enumerate(hams):
        name = f"{i}"
        np.save(folder + f"/{i}", (ham.real).astype(np.float64))
        print(f"save matrix ({ham.shape}): ", folder+ "/" + name + ".npy")