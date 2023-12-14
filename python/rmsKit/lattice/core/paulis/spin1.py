import numpy as np

Sz = np.zeros([3, 3])
Sz[0, 0] = 1
Sz[2, 2] = -1
Sx = np.zeros([3, 3])
Sx[1, 0] = Sx[0, 1] = Sx[2, 1] = Sx[1, 2] = 1/np.sqrt(2)
Sy = np.zeros([3, 3], dtype=np.complex64)
Sy[1, 0] = Sy[2, 1] = 1j/np.sqrt(2)
Sy[0, 1] = Sy[1, 2] = -1j/np.sqrt(2)

Sz = Sz.astype(np.float64)
Sx = Sx.astype(np.float64)
Sy = Sy.astype(np.complex128)

I3 = np.eye(3)

# check spectrul is equal to [-1, 0, 1]
assert np.allclose(np.linalg.eigvalsh(Sz), [-1, 0, 1])
assert np.allclose(np.linalg.eigvalsh(Sx), [-1, 0, 1])
assert np.allclose(np.linalg.eigvalsh(Sy), [-1, 0, 1])

SzSz = np.kron(Sz, Sz).astype(np.float64)
SxSx = np.kron(Sx, Sx).astype(np.float64)
SySy = np.kron(Sy, Sy).real.astype(np.float64)


__all__ = ["SzSz", "SxSx", "SySy", "Sz", "Sx", "Sy"]
