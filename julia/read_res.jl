using Pkg
Pkg.activate("./julia/MinEigLoss")
__precompile__(false)
include("./MinEigLoss/src/MinEigLoss.jl")
include("./MinEigLoss/src/ff.jl")
using JLD2
using LinearAlgebra
using Plots

