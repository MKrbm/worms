using Pkg
Pkg.activate("./MinEigLoss/MinEigLoss")
using MinEigLoss
using SparseArrays
using Arpack
using Zygote    
using LinearAlgebra

d = 3
u = RandomMatrix.Haar(d, Float64)

function n_kron(u::AbstractMatrix{T}, ::Val{n}) where {T <: Real, n}
    us = ntuple(i -> u, Val(n))
    kron(us...)
end

n_kron(u) = n_kron(u, Val(5))

loss(u) = norm(n_kron(u) - I)

Zygote.gradient(loss, u)

y, back = Zygote.pullback(n_kron, u)

back(y)[1]

randn(d^2, d^2) |> vec

function d_n_kron(ȳ, u, ::Val{n}) where n
    g = zero(u)
    ukron = kron(ntuple(i -> u, Val(n-1))...) * n
    d = size(u, 1)
    s = size(ukron, 1)
    for i in 1:d
        for j in 1:d
            for k in 1:s
                for l in 1:s
                    g[i, j] += ukron[k, l] * ȳ[(i-1)*s+k, (j-1)*s+l]
                end
            end
        end
    end
    g
end

using BenchmarkTools
d = 3
y = randn(d, d)
y = n_kron(y, Val(5))

d_n_kron(y, u, Val(5))
back(y)[1]
@benchmark d_n_kron($y, $u, Val(5))
@benchmark back($y)[1]

@code_warntype d_n_kron(y, u, Val(5))

x = randn(10, d, d)
y = randn(d,d)


dot(y, y) == (vec(y)' * vec(y))

d = zero(u)

d[1, 1] = 1
kron(d, u)

