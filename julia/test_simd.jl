using BenchmarkTools

# Function to calculate inner product without SIMD
function inner_product_no_simd(a, b)
    sum = 0.0
    for i in eachindex(a, b)
        sum += a[i] * b[i]
    end
    return sum
end

# Function to calculate inner product with SIMD
function inner_product_simd(a, b)
    sum = 0.0
    @simd for i in eachindex(a, b)
        @inbounds sum += a[i] * b[i]
    end
    return sum
end

# Example vectors
a = rand(10000)
b = rand(10000)

# Benchmarking
no_simd_time = @benchmark inner_product_no_simd($a, $b)
simd_time = @benchmark inner_product_simd($a, $b)

println("No SIMD: ", no_simd_time)
println("With SIMD: ", simd_time)