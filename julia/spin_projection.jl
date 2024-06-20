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

# np = pyimport("numpy");
# ou = np.load("/home/user/project/worms/python/rmsKit/array/torch/FF1D_loc/s_3_r_2_d_1_seed_3/1_mel/Adam/lr_0.01_epoch_100/loss_0.2643156/u/0.npy");
# H = -np.load("/home/user/project/worms/python/rmsKit/array/torch/FF1D_loc/s_$(sd)_r_2_d_1_seed_$(r)/1_mel/H/0.npy");

H = -np.load("/Users/keisuke/Documents/projects/todo/worms/python/rmsKit/array/torch/FF1D_loc/s_16_r_2_d_1_seed_3/1_mel/H/0.npy");
H = Hermitian(H)
# H = randn(sd^2, sd^2)

# SS = spin((1, "xx"), (1, "yy"), (1, "zz"), D=3)
# mat = SS + 1/3 * SS^2
# H = trans_inv_operator(mat, 1:2, L)

# H |> typeof |> fieldnames

# H_sparse = sparse(H)

# λ, U = eigs(H_sparse, which=:SR)

# J2 = (Sx⊗I(3) + I(3)⊗Sx) ^2 + (Sy⊗I(3) + I(3)⊗Sy) ^2 + (Sz⊗I(3) + I(3)⊗Sz) ^2 |> real 

# # eigen(Matrix(J2))
# P = (J2 - 6I) * (J2 - 2I) ./12
# eigen(Matrix(P)).values


function compute_projection(D::Int)
    Px = spin((1, "xI"), (1, "Ix"), D=D)
    Pz = spin((1, "zI"), (1, "Iz"), D=D)
    Py = spin((1, "yI"), (1, "Iy"), D=D)
    PP = Px ^ 2 + Pz ^ 2 + Py ^ 2 |> real

    Λ, U = eigen(Matrix(PP))
    println(Λ)
    spectrum = Λ .|> (u -> round(u, digits = 4)) |> unique
    idx = findall(abs.(spectrum .- spectrum[end - 1]) .< 0.1)
    u = U[:, idx]
    P = sparse(u * u')
    return Matrix(P) |> Hermitian
end

D = 16
# H = compute_projection(D)
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

begin loss = l1; u = rand(Haar(1, D))
    adam = Opt.Adam(u, loss) 
    adam.a = 0.005
    loss_vals_adam_l1 = []
    loss_val_adam_mle = []
    iter = 200
    for i in 1:iter
        Opt.step!(adam)
        push!(loss_vals_adam_l1, l1(adam.theta))
        push!(loss_val_adam_mle, loss_func(adam.theta))
    end
    p1 = plot(1:iter, loss_vals_adam_l1, label="Adam L1")
    p2 = plot(1:iter, loss_val_adam_mle, label="Adam MLE")
    plot(p1, p2, layout=(2,1))
end

begin loss = loss_func; u = rand(Haar(1, D))
    adam = Opt.Adam(u, loss)
    adam.a = 0.005
    sign_bnf = Vector{typeof(u0)}([])
    loss_vals_adam2_l1 = []
    loss_val_adam2_mle = []
    iter = 200
    for i in 1:iter
        if length(sign_bnf) >= 10
            pop!(sign_bnf)
        end
        push!(sign_bnf, (H̄(adam.theta) .|> sign) * (-1) ^ i)
        Opt.step!(adam)
        push!(loss_vals_adam2_l1, l1(adam.theta))
        push!(loss_val_adam2_mle, loss_func(adam.theta))
    end
    p1 = plot(1:iter, loss_vals_adam2_l1, label="Adam2 L1")
    p2 = plot(1:iter, loss_val_adam2_mle, label="Adam2 MLE")
    plot(p1, p2, layout=(2,1))
end

println(loss_val_adam2_mle |> minimum)
println(loss_func(I(D)))

A1 = H̄(rand(Haar(1, D)))
tr(A1 * A1')
A1 .|> (u -> abs(u) ^ 2) |> sum


A1 |> x -> -abs.(x) |> tr
A2 = Ha

A1 ^2 |> tr
A2 ^2 |> tr

round.(A1, digits = 4)
round.(A2, digits = 4)

eigen(Symmetric(A1)).vectors[:, 1]

begin loss = l1; u = rand(Haar(2, D))
    adam = Opt.Adam(u, loss) 
    adam.a = 0.03
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

begin loss = loss_func; u = rand(Haar(2, D))
    adam = Opt.Adam(u, loss) 
    adam.a = 0.01
    loss_vals_adam_l1 = []
    loss_val_adam_mle = []
    iter = 400
    for i in 1:iter
        Opt.step!(adam)
        push!(loss_vals_adam_l1, l1(adam.theta))
        push!(loss_val_adam_mle, loss_func(adam.theta))
    end
    p1 = plot(1:iter, loss_vals_adam_l1, label="Adam L1", title = "Optimize unitary on minimum eigen value")
    p2 = plot(1:iter, loss_val_adam_mle, label="Adam MLE")
    plot(p1, p2, layout=(2,1))
end

loss_func(adam.theta)

adam.theta * adam.theta'

