using Pkg
Pkg.activate("./julia/MinEigLoss")
using MinEigLoss
using ITensors
using Revise



d = 2 # spin degree of freedom
χ = 3 # number of bond dimensions
# i1 = Index(χ, "i1")
# p1 = Index(d, "j1")
# k1 = Index(χ, "k1")
# i2 = Index(χ, "i2")
# j2 = Index(d, "j2")
# k2 = Index(χ, "k2")

ds = Index.(ntuple(x -> d, 4))
χs = Index.(ntuple(x -> χ, 2))
A = randomITensor(ds[1], χs[1], ds[2])
At = randomITensor(ds[3], χs[2], ds[4])
A = A + swapind(A, k1, i1)
At[:, :, :]

At * A * delta(i2, j1)

# C2 = combiner(i2,k2; tags="c2")
# C1 = combiner(i1,k1; tags="c1")
# ci1 = combinedind(C1)
# ci2 = combinedind(C2)
# T = At * A * C1 * C2
# T = Array(T, ci1, ci2)





