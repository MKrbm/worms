using LinearAlgebra
using Zygote
using BenchmarkTools
using Plots

# Function to compute H_plus
function compute_H_plus(A, H)
    abs.(A * H * A')
end

# Function to compute minimum eigenvalue
function min_eigenvalue(H_plus)
    minimum(eigen(H_plus).values)
end

# Function to calculate the minimum eigenvalue with respect to A
function min_eigenvalue_wrt_A(A, H)
    H_plus = compute_H_plus(A, H)
    min_eigenvalue(H_plus)
end

# Different sizes for H
sizes = [10, 20, 50, 100, 200]
backward_times = Float64[]

# Loop over different sizes
for s in sizes
    println("Size: ", s)
    H = Symmetric(rand(s, s))
    A = rand(s, s)
    
    # Measure backward derivative time using Zygote
    backward_time = @belapsed Zygote.gradient(a -> min_eigenvalue_wrt_A(a, H), A)
    push!(backward_times, backward_time)
end



# Plotting the results
plot(sizes, backward_times, label="Zygote (Backward)", xlabel="Matrix Size", ylabel="Time (s)", legend=:topleft, title="Derivative Computational Time")
# savefig("derivative_computational_time.png")

# Display the plot in Julia's plotting window
# display(plot(sizes, backward_times, label="Zygote (Backward)", xlabel="Matrix Size", ylabel="Time (s)", legend=:top_left, title="Derivative Computational Time"))
