using Pkg
Pkg.activate("./julia/MinEigLoss")
using MinEigLoss
using ITensors
using Revise
using LinearAlgebra
using EDKit


d = 3 # spin degree of freedom
χ = 2 # number of bond dimensions

ds = Index.(ntuple(x -> d, 2))
χs = Index.(ntuple(x -> χ, 4))
A = randomITensor(χs[1], ds[1], χs[2])
A = A + swapind(A, χs[1], χs[2])
At = copy(A)
At = swapinds(At, (χs[1], ds[1], χs[2]), (χs[3], ds[2], χs[4]))
T = At * A * delta(χs[2], χs[3])

Di = combiner(ds[1],ds[2]; tags="Di")
Χi = combiner(χs[1],χs[4]; tags="Χi")
di = combinedind(Di)
χi = combinedind(Χi)
E = T * Di * Χi
U, S, V = svd(E, (di))

# Array(prime(U), inds(prime(U)))
# Array(U, inds(U))

# Up = Array(U, inds(U))
# P = I - Up * Up'

# # dag(U) - U
# Up = swapprime(dag(U), 0 => 1) * U * delta(inds(S)[1], inds(S)[1]')
# P = Up - delta(inds(Up))

# L = 4
# H = trans_inv_operator(P |> matrix, 1:2, L)

# bonds = Index.(ntuple(x -> χ, L))
# spins = Index.(ntuple(x -> d, L))

# A_mat = array(A)
# Ms = [itensor(A_mat, bonds[i], spins[i], bonds[mod(i, L) + 1]') for i in 1:L]
# append!(Ms, [delta(bonds[i], bonds[i]') for i in 1:L])
# MPS = reduce(*, Ms)
# P2 = P * dag(Di) * dag(Di')
# MPS*P2*delta(spins[1], ds[1])*delta(spins[2], ds[2]) |> println

P = MinEigLoss.FF.ff(3, 2, dim = Val(1))

H = trans_inv_operator(P, 1:2, 4)