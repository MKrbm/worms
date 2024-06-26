using MinEigLoss
using LinearAlgebra
using Test


function add2(a::Int, b::Int)
    return a + b
end

@testset "randomFloat" begin
    u = RandomMatrix.Haar(5, Float64)
    @test u isa Matrix{Float64}
    @test u * u' ≈ I(5)

    u_mean = [RandomMatrix.Haar(5, Float64) for _ in 1:1_000_000] |> mean |> x -> x./1_000_000

end

@testset "randomComplex" begin
    u = RandomMatrix.Haar(5, ComplexF64)
    @test u isa Matrix{ComplexF64}
    @test u * u' ≈ I(5)
end
