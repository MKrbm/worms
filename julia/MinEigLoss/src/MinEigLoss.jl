module MinEigLoss



using Zygote: @adjoint
using LinearAlgebra
using Zygote
using BenchmarkTools
using Plots
using RandomMatrix

export compute_ground_state, min_eig_loss, min_eigenvalue, H̄, H̄_abs, unitary, riemannian_gradient, riemannian_update

# Function to compute the ground energy and ground state
function compute_ground_state(H::AbstractMatrix)
    E, V = eigen(H)
    ground_energy = minimum(E)
    ground_state = V[:, argmin(E)]
    return ground_energy, ground_state
end

function min_eigenvalue(H::AbstractMatrix)
    minimum(eigen(H).values)
end

function H̄(H, u)
    U = kron(u, u)
    H_sym = U * H * U'
    return Symmetric(H_sym)
end

function H̄_abs(H, u)
    return -abs.(H̄(H, u))
end

function min_eig_loss(H0, u) :: Float64
    u = unitary(u)
    H = H̄_abs(H0, u)
    return -min_eigenvalue(H)
end

# Unitary function and its adjoint
function unitary(u)
    return u
end

@adjoint unitary(u) = unitary(u), c -> (riemannian_gradient(u, c),)

function riemannian_gradient(u, euc_grad)
    rg = euc_grad * u'
    return (rg - rg') / 2
end

function riemannian_update(u, rg, step)
    exp(-step * rg) * u
end

end # module MinEigLoss
