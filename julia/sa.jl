import Pkg
Pkg.activate("julia")
# Pkg.update() #NOTE: If you have not updated the package, you will get an error.
# ENV["PYTHON"] = "/Users/keisuke/miniconda3/envs/torch/bin/python"
# Pkg.build("PyCall")
# using MinEigLoss
using PyCall
using MinEigLoss
using LinearAlgebra 
using RandomMatrix
using Revise
using Zygote
using Plots
using BenchmarkTools    
using SkewLinearAlgebra
using Optim

r = 3
sd = 16
np = pyimport("numpy");
# ou = np.load("/home/user/project/worms/python/rmsKit/array/torch/FF1D_loc/s_3_r_2_d_1_seed_3/1_mel/Adam/lr_0.01_epoch_100/loss_0.2643156/u/0.npy");
H = -np.load("/home/user/project/worms/python/rmsKit/array/torch/FF1D_loc/s_$(sd)_r_2_d_1_seed_$(r)/1_mel/H/0.npy");

H -= eigmax(H) * I;
Ha = Symmetric(H);

H̄(u) = MinEigLoss.H̄(Ha, u);
function wrapper(Ha)
    e0 = eigmin(Ha) |> abs  
    function loss_func(u)
        res = MinEigLoss.min_eig_loss(Ha, u) - e0
        println(res)
        return res
    end
    return loss_func
end

loss_func = wrapper(Ha);

function grad(u :: Matrix{Float64})
    res :: typeof(u) = Zygote.gradient(loss_func, u)[1]
    return res
end

u0 = rand(Haar(1, sd));

function neighbor(u_proposal :: Matrix{Float64}, u_current :: Matrix{Float64})
    retraction = skewhermitian(randn(size(u_proposal)) / sqrt(size(u_proposal)[1]))
    u_proposal[:] = u_current * exp(retraction)
    println(sum(u_proposal))
    return u_proposal
end

sa = Optim.SimulatedAnnealing(
    neighbor = neighbor,
)

res = optimize(loss_func, u0, sa, Optim.Options(iterations=300))
