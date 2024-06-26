using Pkg
Pkg.activate("./MinEigLoss/MinEigLoss")
using MinEigLoss
using Revise
using LinearAlgebra
using EDKit
using Plots
using Zygote
using ITensors
using JLD2



function unitary_optimizer(d, optimizing_loss, tracking_loss; epochs, runs, params, type = :orthogonal)
    best_thetas = []
    best_losses_opt = []
    best_losses_track = []
    function random_theta()
        if type == :orthogonal
            return RandomMatrix.Haar(d, Float64)
        elseif type == :unitary
            return RandomMatrix.Haar(d, ComplexF64)
        else
            throw(ArgumentError("Invalid type: $type"))
        end
    end
    
    for i in 1:runs
        theta = random_theta()
        adam = Opt.Adam(theta, optimizing_loss)
        for (key, value) in params
            setfield!(adam, key, value)
        end
        
        run_best_theta = nothing
        run_best_loss_opt = Inf
        run_best_loss_track = Inf
        
        for j in 1:epochs
            Opt.step!(adam)
            current_loss_opt = optimizing_loss(adam.theta)
            current_loss_track = tracking_loss(adam.theta)
            
            if current_loss_opt < run_best_loss_opt
                run_best_loss_opt = current_loss_opt
                run_best_theta = copy(adam.theta)
                run_best_loss_track = current_loss_track
            end
        end
        
        push!(best_thetas, run_best_theta)
        push!(best_losses_opt, run_best_loss_opt)
        push!(best_losses_track, run_best_loss_track)
    end
    
    return (thetas = best_thetas, losses_opt = best_losses_opt, losses_track = best_losses_track)
end

function special_optimizer(d, optimizing_loss, tracking_loss; epochs, runs, params, type = :orthogonal)
    best_thetas = []
    best_losses_opt = []
    best_losses_track = []
    function random_theta()
        if type == :orthogonal
            w = RandomMatrix.Similarity(d, Float64)
        # elseif type == :unitary
        #     w = randn(ComplexF64, d, d)
        #     w ./= (det(w) |> abs) ^ (1/d)
        else
            throw(ArgumentError("Invalid type: $type"))
        end
        return w
    end
        
    for _ in 1:runs
        theta = random_theta()
        adam = Opt.AdamSpecial(theta, optimizing_loss)
        for (key, value) in params
            setfield!(adam, key, value)
        end
        run_best_theta = nothing
        run_best_loss_opt = Inf
        run_best_loss_track = Inf
        
        for j in 1:epochs
            Opt.step!(adam)
            current_loss_opt = optimizing_loss(adam.theta)
            current_loss_track = tracking_loss(adam.theta)
            
            if current_loss_opt < run_best_loss_opt
                run_best_loss_opt = current_loss_opt
                run_best_theta = copy(adam.theta)
                run_best_loss_track = current_loss_track
            end
        end
        
        push!(best_thetas, run_best_theta)
        push!(best_losses_opt, run_best_loss_opt)
        push!(best_losses_track, run_best_loss_track)
    end
    
    return (thetas = best_thetas, losses_opt = best_losses_opt, losses_track = best_losses_track)
end



function main(seed)
    d = 3; # spin degree of freedom
    χ = 2; # number of bond dimensions
    h = FF.ff(d, χ, dim = Val(1), seed = seed);
    # h = -np.load("/home/user/project/worms/python/rmsKit/array/torch/FF1D_loc/s_3_r_2_d_1_seed_1231/1_mel/H/0.npy");
    L = 4;
    h = h - eigmax(h)I
    H = trans_inv_operator(h, 1:2, L) |> Array;

    mle_unitary = MinEigLoss.mle_unitary(h)
    mle_sys_unitary = MinEigLoss.mle_sys_unitary(H, d)
    mle_sys_special = MinEigLoss.mle_sys_special(H, d)
    l1_sys_special = MinEigLoss.l1_sys_special(H, d)
    mle_special = MinEigLoss.mle_special(h)


    uni_loc = unitary_optimizer(d, mle_unitary, mle_sys_unitary, epochs = 300, runs = 50, params = Dict(:a => 0.02, :b1 => 0.95), type = :unitary);
    uni_loc.losses_opt |> minimum
    uni_loc.losses_track |> minimum

    orth_loc = unitary_optimizer(d, mle_unitary, mle_sys_unitary, epochs = 300, runs = 50, params = Dict(:a => 0.02, :b1 => 0.95), type = :orthogonal);
    orth_loc.losses_opt |> minimum
    orth_loc.losses_track |> minimum

    orth_sys = unitary_optimizer(d, mle_sys_unitary, mle_unitary, epochs = 300, runs = 20, 
                                            params = Dict(:a => 0.02, :b1 => 0.95), type = :orthogonal);

    spec_sys = special_optimizer(d, mle_sys_special, mle_special, epochs = 200, runs = 20, params = Dict(:a => 0.02, :b1 => 0.95), type = :orthogonal);

    return (uni_loc = uni_loc, orth_loc = orth_loc, orth_sys = orth_sys, spec_sys = spec_sys)
end


res = Dict()
for i in 1:1000
    seed = rand(10 ^ 5 : 2 * 10^6)  # Generate random seed between 1000 and 9999
    result = main(seed)
    res[seed] = result
    println("seed: $seed is done")
end
save_object(joinpath(dirname(@__FILE__),"pickles", "res.jld2"), res)
