
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
        return (x * x') / μ
    else
        return abs(x) - μ
    end
end

function Broadcast.broadcasted(::typeof(f), v::Vector{T}) where T
    # Write your custom implementation here to compute the output
    ...
    
    return output
end

map(x -> sabs(x, 1.0), randn(10,10))


x, y = rand(3), rand(3)


x .+ y
