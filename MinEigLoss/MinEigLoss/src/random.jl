module RandomMatrix
    using LinearAlgebra
    function Haar(n::Int, T::Type{<:Number}) :: Matrix{T}
        M = randn(T, n, n)
        Q, R = qr(M)
        Q, R = convert(Matrix,Q),convert(Matrix,R)
        return Q*Diagonal(sign.(diag(R)))
    end

    function Similarity(n::Int, T::Type{<:Number}) :: Matrix{T}
        w = randn(T, n, n)
        w ./= (det(w) |> abs) ^ (1/n)
        return w
    end
end