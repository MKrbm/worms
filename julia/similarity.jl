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

H = -np.load("/home/user/project/worms/python/rmsKit/array/torch/FF1D_loc/s_3_r_2_d_1_seed_3/1_mel/H/0.npy");
D = 3
# H = randn(D^2, D^2) |> Symmetric
H -= eigmax(H) * I;

w = randn(D, D)
w ./= (det(w) |> abs) ^ (1/D)


# MinEigLoss.min_eig_loss(Ha, )
function wrapper_special(Ha)
    e0 = eigmin(Ha) |> abs  
    function loss_func(v)
        return MinEigLoss.min_eig_loss_similarity(Ha, v) - e0
    end
    return loss_func
end

loss_func_special = wrapper_special(H);

function wrapper_unitary(Ha)
    e0 = eigmin(Ha) |> abs  
    function loss_func(u)
        return MinEigLoss.min_eig_loss(Ha, u) - e0
    end
    return loss_func
end

loss_func_unitary = wrapper_unitary(H);

# # W = kron(w,w)
# # W * H * inv(W) |> x -> -abs.(x) |> eigvals

# # H

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

begin
    best_losses_special = []
    best_theta_special = nothing
    min_loss_special = Inf

    for i in 1:100
        begin
            w = randn(D, D)
            w ./= (det(w) |> abs) ^ (1/D)
            adam_l1 = Opt.AdamSpecial(w, loss_func_special)
            adam_l1.a = 0.005
            loss_val_adam_mle = []
            iter = 500
            for j in 1:iter
                Opt.step!(adam_l1)
                current_loss = loss_func_special(adam_l1.theta)
                push!(loss_val_adam_mle, current_loss)
                if current_loss < min_loss_special
                    min_loss_special = current_loss
                    best_theta_special = adam_l1.theta
                end
            end
            p1 = plot(1:iter, loss_val_adam_mle, label="Adam MLE loss = $(loss_val_adam_mle[end])")
            plot(p1, layout=(2,1))
        end
        push!(best_losses_special, loss_val_adam_mle |> minimum)
    end
end

(min_loss_special, best_theta_special)

H |> eigmin
W = kron(best_theta_special, best_theta_special)
W * H * inv(W) .|> (x -> -abs(x)) |> eigvals |> real |> minimum


begin
    best_losses_orthogonal = []
    best_theta_orthogonal = nothing
    min_loss_orthogonal = Inf

    for i in 1:100
        begin
            u = rand(Haar(1, D))
            adam_l1 = Opt.Adam(u, loss_func_unitary)
            adam_l1.a = 0.005
            loss_val_adam_mle = []
            iter = 300
            for j in 1:iter
                Opt.step!(adam_l1)
                current_loss = loss_func_unitary(adam_l1.theta)
                push!(loss_val_adam_mle, current_loss)
                if current_loss < min_loss_orthogonal
                    min_loss_orthogonal = current_loss
                    best_theta_orthogonal = adam_l1.theta
                end
            end
            p1 = plot(1:iter, loss_val_adam_mle, label="Adam MLE loss = $(loss_val_adam_mle[end])")
            plot(p1, layout=(2,1))
        end
        push!(best_losses_orthogonal, loss_val_adam_mle |> minimum)
    end
end

(min_loss_orthogonal, best_theta_orthogonal)

U = kron(best_theta_orthogonal, best_theta_orthogonal)
U * H * U' .|> (x -> -abs(x)) |> eigvals |> real |> minimum



begin
    best_losses_unitary = []
    best_theta_unitary = nothing
    min_loss_unitary = Inf

    for i in 1:100
        begin
            u = rand(Haar(2, D))
            adam_l1 = Opt.Adam(u, loss_func_unitary)
            adam_l1.a = 0.005
            loss_val_adam_mle = []
            iter = 300
            for j in 1:iter
                Opt.step!(adam_l1)
                current_loss = loss_func_unitary(adam_l1.theta)
                push!(loss_val_adam_mle, current_loss)
                if current_loss < min_loss_unitary
                    min_loss_unitary = current_loss
                    best_theta_unitary = adam_l1.theta
                end
            end
            p1 = plot(1:iter, loss_val_adam_mle, label="Adam MLE loss = $(loss_val_adam_mle[end])")
            plot(p1, layout=(2,1))
        end
        push!(best_losses_unitary, loss_val_adam_mle |> minimum)
    end
end

(min_loss_unitary, best_theta_unitary)

U = kron(best_theta_unitary, best_theta_unitary)
U * H * U' .|> (x -> -abs(x)) |> eigvals |> real |> minimum







# Check the system hamiltonian
H
L = 4
Hsys = trans_inv_operator(H, 1:2, L)

eigen(Hsys).values

Hsys |> typeof |> fieldnames

Hsys |> Array |> eigvals

Hsys_sim = trans_inv_operator(H̄s(H, best_theta_special), 1:2, L)

(Hsys_sim |> Array |> eigvals |> real) - (Hsys |> Array |> eigvals |> real) |> norm

Hsys_sim_abs = trans_inv_operator(H̄s_abs(H, best_theta_special), 1:2, L)
Hsys_sim_abs |> Array |> eigvals |> real


function get_system_min_eig(H, L; mode = :normal)
    if mode == :normal
        Hsys = trans_inv_operator(H, 1:2, L) |> Array

    elseif mode == :abs
        Hsys = trans_inv_operator(H, 1:2, L)
        Hsys = -abs.(Hsys |> Array)
    end
    return Hsys |> eigvals |> real |> minimum
end


get_system_min_eig(H̄_abs(H, I(3)), L) # Initial values
get_system_min_eig(H̄s(H, best_theta_special), L, mode = :abs) # Special loss
get_system_min_eig(H̄_abs(H, best_theta_orthogonal), L, mode = :abs) # Orthogonal loss
get_system_min_eig(H̄_abs(H, best_theta_unitary), L, mode = :abs) # Unitary loss


H̄s_abs(H, best_theta_special) |> Array |> eigvals |> real |> minimum
H̄_abs(H, best_theta_unitary) |> Array |> eigvals |> real |> minimum

