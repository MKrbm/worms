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
A = A + swapind(A, k1, i1)
At = copy(A)
At = swapinds(At, (ds[1], χs[1], ds[2]), (ds[3], χs[2], ds[4]))
T = At * A * delta(ds[2], ds[3])

Di = combiner(ds[1],ds[4]; tags="Di")
Χi = combiner(χs[1],χs[2]; tags="Χi")
di = combinedind(Di)
χi = combinedind(Χi)
E = T * Di * Χi
U, S, V = svd(E, (di))

@show S

E = Array(E, di, χi)
