import Pkg
Pkg.activate("julia")
using PyCall
# Pkg.update() #NOTE: If you have not updated the package, you will get an error.
# ENV["PYTHON"] = "/home/user/project/worms/myenv/bin/python"
# Pkg.build("PyCall")
# using MinEigLoss
using MinEigLoss
using LinearAlgebra 
using RandomMatrix
using Revise
using Zygote
using Plots
using BenchmarkTools    
using SkewLinearAlgebra
using SparseArrays
using Arpack
using EDKit

np = pyimport("numpy");

H = -np.load("/home/user/project/worms/python/rmsKit/array/torch/FF1D_loc/s_16_r_2_d_1_seed_3/1_mel/H/0.npy");
H = randn(16, 16) |> Symmetric
H -= eigmax(H) * I;

w = randn(4, 4)
w ./= (det(w) |> abs) ^ (1/4)


# MinEigLoss.min_eig_loss(Ha, )
function wrapper(Ha)
    e0 = eigmin(Ha) |> abs  
    function loss_func(v)
        return MinEigLoss.min_eig_loss_similarity(Ha, v) - e0
    end
    return loss_func
end

loss_func = wrapper(H);

function grad(v :: Matrix{Float64})
    res :: typeof(v) = Zygote.gradient(loss_func, v)[1]
    return res
end

# loss_func(w)

# W = kron(w,w)
# W * H * inv(W) |> x -> -abs.(x) |> eigvals

# H

# begin
#     D = size(w, 1)
#     loss_vals = []
#     for i in 1:1000
#         x = randn(D, D)
#         x = x - tr(x)I / D
#         y = x * w
#         y ./= norm(y, 2)
#         w_prime = riemannian_update(w, y * inv(w), 0.005)
#         push!(loss_vals, loss_func(w_prime))
#     end
# end

# histogram(loss_vals, bins=50)
# rg = grad(w) * w
# rg ./= norm(rg, 2)
# rg = rg * inv(w)
# loss = loss_func(riemannian_update(w, rg, 0.005))
# vline!([loss], color=:red, label="Loss loss = $loss")
# adams = Opt.AdamSpecial(w, loss_func)

# loss_func(w)

