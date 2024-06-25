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

H = -np.load("/home/user/project/worms/python/rmsKit/array/torch/FF1D_loc/s_3_r_2_d_1_seed_1231/1_mel/H/0.npy");
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


function wrapper_unitary(Ha)
    e0 = eigmin(Ha) |> abs  
    function loss_func(u)
        return MinEigLoss.min_eig_loss(Ha, u) - e0
    end
    return loss_func
end



const n_rep = 5
function tuple_w(w) :: NTuple{n_rep, AbstractMatrix}
    ntuple(x -> w, Val(n_rep))
end

@code_warntype tuple_w(w)

function l1_special_wrapper(H:: AbstractMatrix, s_loc :: Int)
    s_sys = size(H, 1)
    n_rep :: Int = log(s_loc, s_sys) |> round |> Int
    function kron_w(w)
        W = kron(w, w)
        for i in 1:n_rep-2
            W = kron(W, w)
        end
        return W
    end

    function wrapper_special(w)
        w = MinEigLoss.special(w)
        W = kron_w(w)
        H̄ = W * H * inv(W)
        H′ = abs.(H̄)
        return sum(H′)
    end
end

function l1_special_wrapper(H:: AbstractMatrix, s_loc :: Int)
    s_sys = size(H, 1)
    n_rep :: Int = log(s_loc, s_sys) |> round |> Int
    function kron_w(w)
        W = kron(w, w)
        for i in 1:n_rep-2
            W = kron(W, w)
        end
        return W
    end

    function wrapper_special(w)
        w = MinEigLoss.special(w)
        W = kron_w(w)
        W_inv = kron_w(inv(w))
        H̄ = W * H * W_inv
        H′ = abs.(H̄)
        return sum(H′)
    end
end

function mle_special_wrapper(H:: AbstractMatrix, s_loc :: Int)
    s_sys = size(H, 1)
    n_rep :: Int = log(s_loc, s_sys) |> round |> Int
    e0 = eigmin(H) |> abs
    function kron_w(w)
        W = kron(w, w)
        for i in 1:n_rep-2
            W = kron(W, w)
        end
        return W
    end

    function wrapper_special(w)
        w = MinEigLoss.special(w)
        W = kron_w(w)
        W_inv = kron_w(inv(w))
        H̄ = W * H * W_inv
        H̄ = abs.(H̄)
        return (eigvals(H̄) |> real |> maximum) - e0
    end
end

function mle_orthogonal_wrapper(H:: AbstractMatrix, s_loc :: Int)
    s_sys = size(H, 1)
    n_rep :: Int = log(s_loc, s_sys) |> round |> Int
    e0 = eigmin(H) |> abs
    function kron_w(w)
        W = kron(w, w)
        for i in 1:n_rep-2
            W = kron(W, w)
        end
        return W
    end

    function wrapper_special(w)
        w = MinEigLoss.unitary(w)
        W = kron_w(w)
        H̄ = W * H * W'
        H̄ = abs.(H̄)
        return (eigvals(H̄) |> real |> maximum) - e0
    end
end
loss_func_special = wrapper_special(H);
loss_func_unitary = wrapper_unitary(H);
L = 4
Hs = trans_inv_operator(H, 1:2, L) |> Array
l1_special = l1_special_wrapper(Hs, size(w, 1))
mle_special = mle_special_wrapper(Hs, size(w, 1))
mle_orthogonal = mle_orthogonal_wrapper(Hs, size(w, 1))


w = randn(D, D)
w ./= (det(w) |> abs) ^ (1/D)
mle_special(w)
loss_func_special(w)
mle_orthogonal(rand(Haar(1, D)))




# begin
#     best_losses_special = []
#     best_theta_special = nothing
#     min_loss_special = Inf

#     for i in 1:100
#         begin
#             w = randn(D, D)
#             w ./= (det(w) |> abs) ^ (1/D)
#             adam_l1 = Opt.AdamSpecial(w, l1_special)
#             adam_l1.a = 0.005
#             loss_val_adam_mle = []
#             loss_val_adam_l1 = []
#             iter = 500
#             for j in 1:iter
#                 Opt.step!(adam_l1)
#                 current_loss = l1_special(adam_l1.theta)
#                 push!(loss_val_adam_l1, current_loss)
#                 push!(loss_val_adam_mle, loss_func_special(adam_l1.theta))
#                 if current_loss < min_loss_special
#                     min_loss_special = current_loss
#                     best_theta_special = adam_l1.theta
#                 end
#             end
#             p1 = plot(1:iter, loss_val_adam_mle, label="Adam MLE loss = $(loss_val_adam_mle[end])")
#             p2 = plot(1:iter, loss_val_adam_l1, label="Adam L1 loss = $(loss_val_adam_l1[end])")
#             plot(p1, p2, layout=(2,1))
#         end
#         push!(best_losses_special, loss_val_adam_mle |> minimum)
#     end
# end

# (min_loss_special, best_theta_special)

# H |> eigmin
# W = kron(best_theta_special, best_theta_special)
# W * H * inv(W) .|> (x -> -abs(x)) |> eigvals |> real |> minimum



begin
    best_losses_special = []
    best_theta_special = nothing
    min_loss_special = Inf

    for i in 1:100
        # begin
        #     w = randn(D, D)
        #     w ./= (det(w) |> abs) ^ (1/D)
        #     adam = Opt.AdamSpecial(w, l1_special)
        #     adam.a = 0.02
        #     adam.b1 = 0.95
        #     loss_val_adam_mle = []
        #     loss_val_adam_mle_local = []
        #     loss_val_adam_l1 = []
        #     iter = 100
        #     for j in 1:iter
        #         Opt.step!(adam)
        #         push!(loss_val_adam_mle_local, loss_func_special(adam.theta))
        #         push!(loss_val_adam_l1, l1_special(adam.theta))
        #         push!(loss_val_adam_mle, mle_special(adam.theta))
        #         if l1_special(adam.theta) < min_loss_special
        #             min_loss_special = l1_special(adam.theta)
        #             best_theta_special = adam.theta
        #         end
        #     end
        #     # p1 = plot(1:iter, loss_val_adam_mle, label="Adam mle system = $(loss_val_adam_mle[end])")
        #     # p2 = plot(1:iter, loss_val_adam_l1, label="Adam l1 system = $(loss_val_adam_l1[end])")
        #     # p3 = plot(1:iter, loss_val_adam_mle_local, label="Adam mle local = $(loss_val_adam_mle_local[end])")
        #     # plot(p1, p2, p3, layout=(3,1))
        # end

        begin
            w = randn(D, D)
            w ./= (det(w) |> abs) ^ (1/D)
            adam = Opt.AdamSpecial(w, mle_special)
            adam.a = 0.01
            adam.b1 = 0.95
            loss_mle = []
            loss_l1 = []
            loss_mle_local = []
            iter = 400
            for j in 1:iter
                Opt.step!(adam)
                push!(loss_mle, mle_special(adam.theta))
                push!(loss_l1, l1_special(adam.theta))
                push!(loss_mle_local, loss_func_special(adam.theta))
            end
            # p1 = plot(1:iter, loss_mle, label="Adam mle system = $(loss_mle[end])")
            # p2 = plot(1:iter, loss_l1, label="Adam l1 system = $(loss_l1[end])")
            # p3 = plot(1:iter, loss_mle_local, label="Adam mle local = $(loss_mle_local[end])")
            # plot(p1, p2, p3, layout=(3,1))
        end
        push!(best_losses_special, loss_mle |> minimum)
    end
end

best_losses_special |> minimum

begin
    best_losses_orthogonal = []
    best_theta_orthogonal = nothing
    min_loss_orthogonal = Inf

    for i in 1:100
        begin
            w = rand(Haar(1, D))
            adam = Opt.Adam(w, mle_orthogonal)
            adam.a = 0.02
            adam.b1 = 0.95
            loss_mle = []
            loss_mle_local = []
            iter = 300
            for j in 1:iter
                Opt.step!(adam)
                current_loss = mle_orthogonal(adam.theta)
                push!(loss_mle, current_loss)
                push!(loss_mle_local, loss_func_unitary(adam.theta))
                if current_loss < min_loss_orthogonal
                    min_loss_orthogonal = current_loss
                    best_theta_orthogonal = adam.theta
                end
            end
            p1 = plot(1:iter, loss_mle, label="Adam mle system = $(loss_mle[end])")
            p2 = plot(1:iter, loss_mle_local, label="Adam mle local = $(loss_mle_local[end])")
            plot(p1, p2, layout=(2,1))
            push!(best_losses_orthogonal, loss_mle |> minimum)
        end
    end
end

best_losses_orthogonal |> minimum

# begin
#     best_losses_unitary = []
#     best_theta_unitary = nothing
#     min_loss_unitary = Inf

#     for i in 1:100
#         begin
#             u = rand(Haar(2, D))
#             adam_l1 = Opt.Adam(u, loss_func_unitary)
#             adam_l1.a = 0.005
#             loss_val_adam_mle = []
#             iter = 300
#             for j in 1:iter
#                 Opt.step!(adam_l1)
#                 current_loss = loss_func_unitary(adam_l1.theta)
#                 push!(loss_val_adam_mle, current_loss)
#                 if current_loss < min_loss_unitary
#                     min_loss_unitary = current_loss
#                     best_theta_unitary = adam_l1.theta
#                 end
#             end
#             p1 = plot(1:iter, loss_val_adam_mle, label="Adam MLE loss = $(loss_val_adam_mle[end])")
#             plot(p1, layout=(2,1))
#         end
#         push!(best_losses_unitary, loss_val_adam_mle |> minimum)
#     end
# end

# (min_loss_unitary, best_theta_unitary)

# U = kron(best_theta_unitary, best_theta_unitary)
# U * H * U' .|> (x -> -abs(x)) |> eigvals |> real |> minimum


# begin
#     w = randn(D, D)
#     w ./= (det(w) |> abs) ^ (1/D)
#     adam_l1 = Opt.AdamSpecial(w, l1_special)
#     adam_l1.a = 0.005
#     loss_val_adam_l1 = []
#     loss_val_adam_mle = []
#     iter = 500
#     for j in 1:iter
#         Opt.step!(adam_l1)
#         current_loss = loss_func_special(adam_l1.theta)
#         push!(loss_val_adam_mle, current_loss)
#         jush!(loss_val_adam_l1, l1_special(adam_l1.theta))
#         if current_loss < min_loss_special
#             min_loss_special = current_loss
#             best_theta_special = adam_l1.theta
#         end
#     end
#     p1 = plot(1:iter, loss_val_adam_mle, label="Adam MLE loss = $(loss_val_adam_mle[end])")
#     p2 = plot(1:iter, loss_val_adam_l1, label="Adam L1 loss = $(loss_val_adam_l1[end])")
#     plot(p1, p2, layout=(2,1))
# end





# # Check the system hamiltonian
# H
# L = 4
# Hsys = trans_inv_operator(H, 1:2, L)

# eigen(Hsys).values

# Hsys |> typeof |> fieldnames

# Hsys |> Array |> eigvals

# Hsys_sim = trans_inv_operator(H̄s(H, best_theta_special), 1:2, L)

# (Hsys_sim |> Array |> eigvals |> real) - (Hsys |> Array |> eigvals |> real) |> norm

# Hsys_sim_abs = trans_inv_operator(H̄s_abs(H, best_theta_special), 1:2, L)
# Hsys_sim_abs |> Array |> eigvals |> real


# function get_system_min_eig(H, L; mode = :normal)
#     if mode == :normal
#         Hsys = trans_inv_operator(H, 1:2, L) |> Array

#     elseif mode == :abs
#         Hsys = trans_inv_operator(H, 1:2, L)
#         Hsys = -abs.(Hsys |> Array)
#     end
#     println(Hsys |> size)
#     return Hsys |> eigvals |> real |> minimum
# end


# get_system_min_eig(H̄_abs(H, I(3)), L) # Initial values
# get_system_min_eig(H̄s(H, best_theta_special), L, mode = :abs) # Special loss
# get_system_min_eig(H̄_abs(H, best_theta_orthogonal), L, mode = :abs) # Orthogonal loss
# get_system_min_eig(H̄_abs(H, best_theta_unitary), L, mode = :abs) # Unitary loss


# H̄s_abs(H, best_theta_special) |> Array |> eigvals |> real |> minimum
# H̄_abs(H, best_theta_unitary) |> Array |> eigvals |> real |> minimum


# Hs = H̄s(H, best_theta_special);
# Hs_abs = H̄s_abs(H, best_theta_special)
# # Hs = H̄(H, best_theta_unitary)
# I3 = I(3);

# Hsys = kron(Hs_abs, I3, I3)  + kron(I3, Hs_abs, I3)

# # Hsys = Hs
# Hsys |> eigvals
# Hsys |> (x -> -abs.(x)) |> eigvals 




