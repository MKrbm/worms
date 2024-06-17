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
ou = np.load("/home/user/project/worms/python/rmsKit/array/torch/FF1D_loc/s_3_r_2_d_1_seed_3/1_mel/Adam/lr_0.01_epoch_100/loss_0.2643156/u/0.npy")
const H = -np.load("/home/user/project/worms/python/rmsKit/array/torch/FF1D_loc/s_3_r_2_d_1_seed_3/1_mel/H/0.npy")
H -= eigmax(H) * I
# const Ha :: Hermitian{Float64} = hermitianpart(H)
Ha = hermitianpart(H)
const e0 = eigmin(Ha) |> abs

H̄(u) = MinEigLoss.H̄(Ha, u)

function wrapper(Ha)
    e0 = eigmin(Ha) |> abs  
    println(e0)
    function loss_func(u)
        return MinEigLoss.min_eig_loss(Ha, u) - e0
    end
    return loss_func
end

loss_func = wrapper(Ha)
@code_warntype loss_func(ou)


function riemannian_steepest_descent(loss :: Function, u0 :: Matrix{Float64}, step :: Float64; iterations :: Int = 1000) 
    losses = Float64[]
    rgs = Matrix{Float64}[]
    u = copy(u0)
    for i in 1:iterations
        l = loss(u)
        push!(losses, l)
        rg :: typeof(u) = Zygote.gradient(loss, u)[1]
        push!(rgs, rg)
        if !(norm(rg + rg') < 1e-7)
            println("gradient is not skew-symmetric")
            break
        end
        u = copy(MinEigLoss.riemannian_update(u, rg, step))
    end

    return losses, u, rgs
end



iter = 3000
step = 1e-3
u0 = rand(Haar(1, 3));
losses, final_u, rgs = riemannian_steepest_descent(loss_func, u0, step; iterations = iter);
println(losses[end])
@code_warntype riemannian_steepest_descent(loss_func,u0, step; iterations = iter)

plot(1:iter, losses, xlabel="Iteration", ylabel="Loss", title="Riemannian Steepest Descent Loss", legend=false)



rg = Zygote.gradient(loss_func, ou)[1]
I_matrix = I(size(rg)[1])

H_old = zero(H);
function sign_change(H_new)
    global H_old
    res = abs.(sign.(H_new) - sign.(H_old))  |> sum
    H_old = H_new
    return res
end

function landscape(t, r, u, q)
    uΔt = exp(t .* r) * u
    H_new = H̄(uΔt) |> Symmetric
    # println(H_new)
    # lq_norm_res = lq_norm(H_new, q)
    # println(typeof(lq_norm_res))
    return loss_func(uΔt), sum_near_zero(uΔt, 0.01), sign_change(H_new), lq_norm(H_new, q)
end

function sum_near_zero(x, step)
    return sum(abs.(H̄(x)) .< step)
end

function lq_norm(x :: AbstractMatrix, q)
    loss = zero(eltype(x))
    s = 9
    println(size(x))
    for i in 1:s
        for j in 1:s
            if i == j
                continue
            end
            loss += abs(x[i,j]) ^ q
        end
    end
    return loss
end

let u = final_u
    rg = Zygote.gradient(loss_func, u)[1]
    t_list = LinRange(-10, 10, 1000)
    unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))
    loss_vals, near_zeros, sign_changes, lq_norms = unzip([landscape(t, rg, u, 1) for t in t_list])
    sign_changes[1] = 0
    p1 = plot(t_list, loss_vals, xlabel="Δt", ylabel="Loss", title="Riemannian Steepest Descent Loss", legend=false)
    p2 = plot(t_list, near_zeros, color="red", xlabel="Δt", ylabel="Near Zero", title="Near Zero", legend=false)
    p3 = plot(t_list, sign_changes, color="blue", xlabel="Δt", ylabel="Sign Change", title="Sign Change", legend=false,)
    p4 = plot(t_list, lq_norms, color="green", xlabel="Δt", ylabel="LQ Norm", title="LQ Norm", legend=false)

    loss_vals |> minimum
    plot(p1, p2, p3, p4, layout=(4,1))
end




