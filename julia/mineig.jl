using Zygote: @adjoint
using LinearAlgebra
using Zygote
using BenchmarkTools
using Plots
using RandomMatrix

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
    H_sym = abs.(U * H * U')
    return Symmetric(H_sym)
end

function loss(H0, u) :: Float64
    u = unitary(u)
    H = H̄(H0, u)
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

# Function to perform Riemannian steepest descent and plot the loss
function riemannian_steepest_descent(H, u0, step, iterations)
    losses = Float64[]
    rgs = Matrix{Float64}[]
    u = u0

    for i in 1:iterations
        l = loss(u)
        push!(losses, l)
        rg = Zygote.gradient(u -> loss(H, u), u)[1]
        push!(rgs, rg)
        if !(norm(rg + rg') < 1e-7)
            println("gradient is not skew-symmetric")
            break
        end
        u = riemannian_update(u, rg, step)
    end

    return losses, u, rgs
end

# Example usage
const H = Symmetric(rand(64, 64));
u0 = rand(Haar(1, 8));
step = 0.01

loss(u) = loss(H, u)

f(u) = loss(H,u)
Zygote.gradient(f, u0);

iterations = 1000

losses, final_u, rgs = riemannian_steepest_descent(H,u0, step, iterations);
plot(1:iterations, losses, xlabel="Iteration", ylabel="Loss", title="Riemannian Steepest Descent Loss", legend=false)


# Plotting along with rg randomly selected from rgs


# rg = rgs[rand(1:iterations)]
rg = rgs[end]
x = range(-1, stop=1, length=100)
u_init = final_u
losses_rg = Float64[]
for val in x
    u = riemannian_update(u_init, rg, val)
    l = loss(H, u)
    push!(losses_rg, l)
end

losses
p1 = plot(x, losses_rg, xlabel="Step Size", ylabel="Loss", title="Loss vs Step Size", legend=false)

rg_flat = vec(rg)
random_vec = randn(length(rg_flat))
p_vec = random_vec - (dot(random_vec, rg_flat) / dot(rg_flat, rg_flat)) * rg_flat
p_rg = reshape(p_vec, size(rg))
p_rg = p_rg - p_rg'

# Draw the landscape of loss for this value
x = range(-3, stop=1, length=100)
u_init = final_u
losses_p_rg = Float64[]
for val in x
    u = riemannian_update(u_init, p_rg, val)
    l = loss(H, u)
    push!(losses_p_rg, l)
end

p2 = plot(x, losses_p_rg, xlabel="Step Size", ylabel="Loss", title="Loss vs Step Size (Perpendicular rg)", legend=false)

plot(p1, p2)


typeof(3) |> isabstracttype
Real |> isabstracttype