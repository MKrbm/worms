import Pkg
Pkg.activate("julia")
# Pkg.update() #NOTE: If you have not updated the package, you will get an error.
ENV["PYTHON"] = "/Users/keisuke/miniconda3/envs/torch/bin/python"
Pkg.build("PyCall")
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

r = 3
sd = 16
np = pyimport("numpy");
# ou = np.load("/home/user/project/worms/python/rmsKit/array/torch/FF1D_loc/s_3_r_2_d_1_seed_3/1_mel/Adam/lr_0.01_epoch_100/loss_0.2643156/u/0.npy");
# H = -np.load("/home/user/project/worms/python/rmsKit/array/torch/FF1D_loc/s_$(sd)_r_2_d_1_seed_$(r)/1_mel/H/0.npy");
H = -np.load("/Users/keisuke/Documents/projects/todo/worms/python/rmsKit/array/torch/FF1D_loc/s_16_r_2_d_1_seed_3/1_mel/H/0.npy");

# H = randn(sd^2, sd^2)
# H = (H + H') / 2
H -= eigmax(H) * I;
Ha = Symmetric(H);

# Ha = Symmetric(randn(sd^2, sd^2))


H̄(u) = MinEigLoss.H̄(Ha, u);
# MinEigLoss.min_eig_loss(Ha, )
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

u0 = rand(Haar(1, sd));

begin   
    adam = Opt.Adam(u0, loss_func)
    sg = Opt.SG(u0, loss_func)
    sg.a = 0.01
    adam.a = 0.01
    loss_vals_adam = []
    loss_vals_sg = []
    iter = 100
    for i in 1:iter
        Opt.step!(adam)
        Opt.step!(sg)
        push!(loss_vals_adam, adam.loss(adam.theta))
        push!(loss_vals_sg, sg.loss(sg.theta))
    end

    p = plot(1:iter, loss_vals_adam, label="Adam")
    plot(p, 1:iter, loss_vals_sg, label="SG")
end

function lq_norm(x :: AbstractMatrix, q)
    loss = zero(eltype(x))
    s = size(x)[1]
    x_abs = abs.(x)
    loss = sum(x_abs .^ q)
    return loss ^ (1/q)
end

function reg(u :: AbstractMatrix)
    u′ = MinEigLoss.unitary(u)
    return lq_norm(H̄(u′), 1) 
end

u0 = rand(Haar(1, sd))

adam_l1 = Opt.Adam(u0, reg)
adam_l1.a = 0.01
loss_vals_adam_l1 = []
loss_val_adam_mle = []
iter = 50
begin 
    adam_l1 = Opt.Adam(u0, reg)
    adam_l1.a = 0.01
    loss_vals_adam_l1 = []
    loss_val_adam_mle = []
    for i in 1:iter
        Opt.step!(adam_l1)
        push!(loss_vals_adam_l1, reg(adam_l1.theta))
        push!(loss_val_adam_mle, loss_func(adam_l1.theta))
    end
    p1 = plot(1:iter, loss_vals_adam_l1, label="Adam L1")
    p2 = plot(1:iter, loss_val_adam_mle, label="Adam MLE")
    plot(p1, p2, layout=(2,1))
end


begin
    adam_second = Opt.Adam(adam_l1.theta, loss_func)
    adam_second.a = 0.005
    loss_vals_adam_l1 = []
    loss_val_adam_mle = []
    iter = 50
    for i in 1:iter
        Opt.step!(adam_second)
        push!(loss_vals_adam_l1, reg(adam_second.theta))
        push!(loss_val_adam_mle, loss_func(adam_second.theta))
    end
    p1 = plot(1:iter, loss_vals_adam_l1, label="Adam L1")
    p2 = plot(1:iter, loss_val_adam_mle, label="Adam MLE")
    plot(p1, p2, layout=(2,1))
end

H_old = zero(Ha);
function sign_change(H_new)
    global H_old
    res = abs.(sign.(H_new) - sign.(H_old))  |> sum
    H_old = H_new
    return res
end

function landscape(t, r, u, q)
    uΔt = exp(t .* r) * u
    H_new = H̄(uΔt) |> Symmetric
    return loss_func(uΔt), sum_near_zero(uΔt, 0.01), sign_change(H_new), reg(uΔt)
end

function sum_near_zero(x, step)
    return sum(abs.(H̄(x)) .< step)
end


let u = adam_second.theta
    rg = grad(u)
    rg /= norm(rg)
    t_list = LinRange(-10, 10, 1000)
    unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))
    loss_vals, near_zeros, sign_changes, lq_norms = unzip([landscape(t, rg, u, 1) for t in t_list])
    sign_changes[1] = 0
    p1 = plot(t_list, loss_vals, xlabel="Δt", ylabel="Loss", title="Riemannian Steepest Descent Loss", legend=false)
    p2 = plot(t_list, near_zeros, color="red", xlabel="Δt", ylabel="Near Zero", title="Near Zero", legend=false)
    p3 = plot(t_list, sign_changes, color="blue", xlabel="Δt", ylabel="Sign Change", title="Sign Change", legend=false,)
    p4 = plot(t_list, lq_norms, color="green", xlabel="Δt", ylabel="LQ Norm", title="LQ Norm", legend=false)

    loss_vals |> minimum
    plot(p1, p2, p3, p4, layout=(4,1), size=(800, 1200))
end




let u = adam_second.theta
    rg = grad(u);
    rg /= norm(rg)  
    u1 = reshape(rg, :);
    w2 = randn(size(u1));
    u2 = w2 - (u1'w2)/(u1'u1) * u1;
    rg2 = reshape(u2, size(rg));
    rg2  = rg2 - rg2';
    rg /= norm(rg2);
    t_list = LinRange(-10, 10, 1000)
    unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))
    loss_vals, near_zeros, sign_changes, lq_norms = unzip([landscape(t, rg, u, 1) for t in t_list])
    sign_changes[1] = 0
    p1 = plot(t_list, loss_vals, xlabel="Δt", ylabel="Loss", title="Riemannian Steepest Descent Loss", legend=false)
    p2 = plot(t_list, near_zeros, color="red", xlabel="Δt", ylabel="Near Zero", title="Near Zero", legend=false)
    p3 = plot(t_list, sign_changes, color="blue", xlabel="Δt", ylabel="Sign Change", title="Sign Change", legend=false,)
    p4 = plot(t_list, lq_norms, color="green", xlabel="Δt", ylabel="LQ Norm", title="LQ Norm", legend=false)

    loss_vals |> minimum
    plot(p1, p2, p3, p4, layout=(4,1), size=(800, 1200))
end

