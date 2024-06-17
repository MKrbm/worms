using Zygote: @adjoint
using LinearAlgebra
using Zygote
using BenchmarkTools
using Plots

# Function to compute H_plus
function compute_H_plus(A, H)
    abs.(A * H * A')
end

# Function to compute minimum eigenvalue
function min_eigenvalue(H_plus)
    minimum(eigen(H_plus).values)
end

# Function to calculate the minimum eigenvalue with respect to A
function min_eigenvalue_wrt_A(A, H)
    H_plus = compute_H_plus(A, H)
    min_eigenvalue(H_plus)
end



A = Hermitian(randn(ComplexF64, 10, 10));
y, back = Zygote.pullback(min_eigenvalue, A)
grad = Zygote.gradient(min_eigenvalue, A)

back(1.0)

# Gradient descent parameters
learning_rate = 0.01
num_steps = 100

# Initial matrix A
# A = randn(Float64, 10, 10)
H = Symmetric(randn(Float64, 10, 10))  # Assuming H is also Hermitian

(E, V) = eigen(H)
grad = Zygote.gradient(min_eigenvalue, H)
grad
E[1]
minimum(E)
V[:, 1]' * H * V[:, 1]

function grad_min_eigenvalue(c)
    v = V[:, 1]::Vector{Float64}
    c * v * v'
end


# g = c -> (grad_min_eigenvalue(c),)
# grad_min_eigenvalue(H)

# @code_warntype g(H)

# @adjoint min_eigenvalue(H) = min_eigenvalue(H), c -> (grad_min_eigenvalue(c),)

# grad2 = Zygote.gradient(min_eigenvalue, H)

# grad2[1]

# grad[1] - grad2[1]


# Define the SymmetryEigen type
struct SymmetryEigen{T<:AbstractMatrix}
    matrix::T
    ground_energy::Float64
    ground_state::Vector{Float64}
end

# Function to compute the ground energy and ground state
function compute_ground_state(H::AbstractMatrix)
    E, V = eigen(H)
    ground_energy = minimum(E)
    ground_state = V[:, argmin(E)]
    return ground_energy, ground_state
end

# Constructor for the SymmetryEigen type
function SymmetryEigen(H::T) where {T<:AbstractMatrix}
    ground_energy, ground_state = compute_ground_state(H)
    return SymmetryEigen(H, ground_energy, ground_state)
end

function VariationEigen(H::SymmetryEigen)
    v = H.ground_state
    v' * H.matrix * v
end

function Lanczos(H::SymmetryEigen)
    v = H.ground_state
    v' * H.matrix * v
end


H_sym = Symmetric(randn(Float64, 64, 64))
sym_eigen_sym = SymmetryEigen(H_sym);

VariationEigen(sym_eigen_sym);
sym_eigen_sym.ground_energy;

@benchmark eigen($H_sym, 1:1)
@benchmark eigen($H_sym)

function inverse_iteration(H::SymmetryEigen, mu::Real, tol::Real, max_iter::Int)
    mu = H.ground_energy - mu
    A = H.matrix - mu * I
    v = H.ground_state
    v /= norm(v)    # Normalize the initial vector
    for i in 1:max_iter
        w = A \ v  # Solve (A - mu*I)w = v
        v_n = w / norm(w)       # Normalize the vector
        # Check for convergence
        if norm(v_n - v) < tol
            return mu, v
        end
        v = v_n
    end
    println("Failed to converge within the maximum number of iterations.")
    return mu, v
end
@benchmark inverse_iteration(sym_eigen_sym, 0.01, 1e-7, 10)

function min_eigenvalue(H_plus)
    minimum(eigen(H_plus).values)
end

function mul_y(y)
    println("y = $y")
    return y*y
end

function two_argument(x, y)
    c = mul_y(y)
    return x + c
end

@code_warntype two_argument(3.0, 3.0)
two_argument(3.0, 3.0)

const y = 3.0
one_arg(x) = two_argument(x, y)

one_arg(3.0)

@code