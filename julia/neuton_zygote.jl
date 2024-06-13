import Pkg
Pkg.activate("julia")
# ENV["PYTHON"] = "/Users/keisuke/miniconda3/envs/torch/bin/python"
# Pkg.build("PyCall")
# using MinEigLoss
using PyCall
import MinEigLoss
using LinearAlgebra 
using RandomMatrix
using Revise
using Zygote
using Plots

np = pyimport("numpy")
BASE_DIR = "/Users/keisuke/Documents/projects/todo/worms/job/FF1D_sps_8/FF1D"
BASE_DIR *= "visualize/plots/compare_o_vs_u/sps_8" 
# FIGURE_DIR = joinpath(BASE_DIR, "visualize", "plots", "compare_o_vs_u", "sps_8")
ou = np.load("/Users/keisuke/Documents/projects/todo/worms/python/rmsKit/array/torch/FF1D_loc/s_3_r_2_d_1_seed_3/1_mel/Adam/lr_0.01_epoch_100/loss_0.2644055/u/0.npy")
H = -np.load("/Users/keisuke/Documents/projects/todo/worms/python/rmsKit/array/torch/FF1D_loc/s_3_r_2_d_1_seed_3/1_mel/H/0.npy")
H -= eigmax(H) * I
e0 = eigmin(H) |> abs

u0 = rand(Haar(1, 8));

MinEigLoss.min_eig_loss(H, ou) - e0

Zygote.gradient(loss, u0)

H̄(u) = MinEigLoss.H̄(H, u)

function sum_near_zero(x, step)
    return sum(abs.(H̄(x)) .< step)
end

function riemannian_steepest_descent(H, u0, step; iterations = 1000)
    losses = Float64[]
    rgs = Matrix{Float64}[]
    u = u0
    loss(u) = MinEigLoss.min_eig_loss(H, u)

    for i in 1:iterations
        l = loss(u)
        push!(losses, l)
        rg = Zygote.gradient(loss, u)[1]
        push!(rgs, rg)
        if !(norm(rg + rg') < 1e-7)
            println("gradient is not skew-symmetric")
            break
        end
        u = MinEigLoss.riemannian_update(u, rg, step)
    end

    return losses, u, rgs
end

iter = 1000
step = 1e-3
losses, final_u, rgs = riemannian_steepest_descent(H,u0, step, iterations = iter);


plot(1:iter, losses, xlabel="Iteration", ylabel="Loss", title="Riemannian Steepest Descent Loss", legend=false)


rg = Zygote.gradient(loss, ou)[1]
I_matrix = I(size(rg)[1])

H_old = zero(H);
function sign_change(H_new)
    global H_old
    res = abs.(sign.(H_new) - sign.(H_old))  |> sum
    H_old = H_new
    return res
end

function landscape(t, r, u)
    uΔt = exp(t .* r) * u
    return loss(uΔt), sum_near_zero(uΔt, 0.005), sign_change(H̄(uΔt))
end




t_list = LinRange(-10, 10, 100)
unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))
loss_vals, near_zeros, sign_changes = unzip([landscape(t, rg, ou) for t in t_list])
sign_changes[1] = 0
p1 = plot(t_list, loss_vals, xlabel="Δt", ylabel="Loss", title="Riemannian Steepest Descent Loss", legend=false)
p2 = plot(t_list, near_zeros, color="red", xlabel="Δt", ylabel="Near Zero", title="Near Zero", legend=false)
p3 = plot(t_list, sign_changes, color="blue", xlabel="Δt", ylabel="Sign Change", title="Sign Change", legend=false,)


plot(p1, p2, p3, layout=(3,1))


