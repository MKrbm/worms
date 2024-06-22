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


function sabs(x ::Number, μ ::Float64)
    if abs(x) < μ
        return (x * x') / (2μ) + μ / 2
    else
        return abs(x)
    end
end

sabs(x :: Number) = sabs(x, 1.0)

sabs.(randn(10))

function Broadcast.broadcasted(::typeof(f), v::Vector{T}) where T
    # Write your custom implementation here to compute the output
    ...
    
    return output
end

t_list = LinRange(-1, 1, 100)
plot(t_list, t_list .|> u -> sabs(u, 0.1))
plot!(t_list, t_list .|> u -> abs(u))