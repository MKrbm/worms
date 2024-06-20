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

r = 3
sd = 16
np = pyimport("numpy");
# ou = np.load("/home/user/project/worms/python/rmsKit/array/torch/FF1D_loc/s_3_r_2_d_1_seed_3/1_mel/Adam/lr_0.01_epoch_100/loss_0.2643156/u/0.npy");
H = -np.load("/Users/keisuke/Documents/projects/todo/worms/python/rmsKit/array/torch/FF1D_loc/s_16_r_2_d_1_seed_3/1_mel/H/0.npy");
# H = -np.load("/home/user/project/worms/python/rmsKit/array/torch/FF1D_loc/s_16_r_2_d_1_seed_3/1_mel/H/0.npy");

# H = randn(sd^2, sd^2)
H = Hermitian(H)
H -= eigmax(H) * I;
Ha = Hermitian(H);

H̄(u) = MinEigLoss.H̄(Ha, u);
H̄_abs(u) = MinEigLoss.H̄_abs(Ha, u)
function wrapper(Ha)
    e0 = eigmin(Ha) |> abs  
    function loss_func(u)
        return MinEigLoss.min_eig_loss(Ha, u) - e0
    end
    return loss_func
end

loss_func = wrapper(Ha);

function grad(u :: Matrix{Float64})
    res :: typeof(u) = Zygote.gradient(loss_func, u)[1]
    return res
end

function l1(u :: AbstractMatrix)
    u′ = MinEigLoss.unitary(u)
    H′ = abs.(H̄(u′))
    return sum(H′)
end

u0 = rand(Haar(1, sd))
begin loss = l1
    adam_l1 = Opt.Adam(u0, loss) 
    adam_l1.a = 0.05
    loss_vals_adam_l1 = []
    loss_val_adam_mle = []
    iter = 100
    for i in 1:iter
        Opt.step!(adam_l1)
        push!(loss_vals_adam_l1, l1(adam_l1.theta))
        push!(loss_val_adam_mle, loss_func(adam_l1.theta))
    end
    p1 = plot(1:iter, loss_vals_adam_l1, label="Adam L1", title = "Optimize unitary on L1 norm")
    p2 = plot(1:iter, loss_val_adam_mle, label="Adam MLE")
    plot(p1, p2, layout=(2,1))
end


begin loss = loss_func; u = adam_l1.theta
    adam2 = Opt.Adam(u, loss)
    adam2.a = 0.05
    sign_bnf = Vector{typeof(u0)}([])
    loss_vals_adam2_l1 = []
    loss_val_adam2_mle = []
    iter = 50
    for i in 1:iter
        if length(sign_bnf) >= 10
            pop!(sign_bnf)
        end
        push!(sign_bnf, (H̄(adam2.theta) .|> sign) * (-1) ^ i)
        Opt.step!(adam2)
        push!(loss_vals_adam2_l1, l1(adam2.theta))
        push!(loss_val_adam2_mle, loss_func(adam2.theta))
    end
    p1 = plot(1:iter, loss_vals_adam2_l1, label="Adam2 L1", title = "Optimize orthogonal on minimumeigenloss norm")
    p2 = plot(1:iter, loss_val_adam2_mle, label="Adam2 MLE")
    plot(p1, p2, layout=(2,1))
end

begin loss = l1; u = rand(Haar(2, sd))
    adam = Opt.Adam(u, loss) 
    adam.a = 0.05
    loss_vals_adam_l1 = []
    loss_val_adam_mle = []
    iter = 100
    for i in 1:iter
        Opt.step!(adam)
        push!(loss_vals_adam_l1, l1(adam.theta))
        push!(loss_val_adam_mle, loss_func(adam.theta))
    end
    p1 = plot(1:iter, loss_vals_adam_l1, label="Adam L1", title = "Optimize unitary on L1 norm")
    p2 = plot(1:iter, loss_val_adam_mle, label="Adam MLE")
    plot(p1, p2, layout=(2,1))
end

begin loss = loss_func; u = adam.theta
    adam = Opt.Adam(u, loss) 
    adam.a = 0.04
    loss_vals_adam_l1 = []
    loss_val_adam_mle = []
    iter = 100
    for i in 1:iter
        Opt.step!(adam)
        push!(loss_vals_adam_l1, l1(adam.theta))
        push!(loss_val_adam_mle, loss_func(adam.theta))
    end
    p1 = plot(1:iter, loss_vals_adam_l1, label="Adam L1", title = "Optimize unitary on minimum eigen value")
    p2 = plot(1:iter, loss_val_adam_mle, label="Adam MLE")
    plot(p1, p2, layout=(2,1))
end


@code_warntype Opt.step!(adam)
