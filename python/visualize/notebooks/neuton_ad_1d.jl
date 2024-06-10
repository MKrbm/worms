using LinearAlgebra
using ForwardDiff: gradient, hessian
using Optim
using Plots

# Define the 1D objective function
function f(x::AbstractVector{<:Real})
    return f(x[1])
end

f(x::Real) = abs(cos(x))

# Define gradient and Hessian
grad_f = x -> gradient(f, x)
hess_f = x -> hessian(f, x)

# Newton's method
function newton_method(f, x0; tol=1e-6, max_iter=100)
    x = x0
    x_vals = [x0[1]]
    for i in 1:max_iter
        g = grad_f(x)
        h = hess_f(x)
        step = -h \ g
        x = x + step
        push!(x_vals, x[1])
        if norm(g) < tol
            return x, i, x_vals
        end
    end
    return x, max_iter, x_vals
end

# Damped Newton's method
function damped_newton_method(f, x0; tol=1e-6, max_iter=100, alpha=0.5)
    x = x0
    x_vals = [x0[1]]
    for i in 1:max_iter
        g = grad_f(x)
        h = hess_f(x)
        step = -h \ g
        x = x + alpha * step
        push!(x_vals, x[1])
        if norm(g) < tol
            return x, i, x_vals
        end
    end
    return x, max_iter, x_vals
end

# Quasi-Newton method using BFGS with callback
function quasi_newton_bfgs(f, x0; tol=1e-6, max_iter=100)
    x_vals = []
    callback = (p) -> begin
        push!(x_vals, p.metadata["x"])
        false
    end
    options = Optim.Options(callback = callback, show_trace = false, extended_trace = true)
    result = optimize(f, x0, BFGS(), options)
    # println(options)
    # println(result)
    return Optim.minimizer(result), Optim.iterations(result), x_vals
end

# Initial guess
x0 = [0.2]

# Run the methods
x_newton, iter_newton, x_vals_newton = newton_method(f, x0)
x_damped_newton, iter_damped_newton, x_vals_damped_newton = damped_newton_method(f, x0)
x_bfgs, iter_bfgs, x_vals_bfgs = quasi_newton_bfgs(f, x0)
x_vals_bfgs = Float64[x[1] for x in x_vals_bfgs];
x_vals_damped_newton
x_vals_newton

# Print results
println("Newton's method: x = $x_newton, iterations = $iter_newton")
println("Damped Newton's method: x = $x_damped_newton, iterations = $iter_damped_newton")
println("Quasi-Newton (BFGS) method: x = $x_bfgs, iterations = $iter_bfgs")

# Create a grid for the landscape of the target function
xgrid = range(-1, 6, length=300)
z = [f([x]) for x in xgrid]

# Create the plot for the landscape of the target function with optimization paths
p = plot(xgrid, z, linewidth=1.5, xlabel="x-axis", ylabel="y-axis", title="Landscape of Target Function with Optimization Paths", legend=:topright)
scatter!(p, x_vals_newton, [f([x]) for x in x_vals_newton], label="Newton's method", color=:red, markersize=3, marker=:circle)
scatter!(p, x_vals_damped_newton, [f([x]) for x in x_vals_damped_newton], label="Damped Newton's method", color=:blue, markersize=3, marker=:square)
scatter!(p, x_vals_bfgs, [f([x]) for x in x_vals_bfgs], label="Quasi-Newton (BFGS) method", color=:green, markersize=3, marker=:diamond)

# Create the second subplot for the optimization paths
p2 = plot(1:length(x_vals_newton), x_vals_newton, label="Newton's method", xlabel="Iteration", ylabel="x value", title="Optimization Paths")
plot!(p2, 1:length(x_vals_damped_newton), x_vals_damped_newton, label="Damped Newton's method")
plot!(p2, 1:length(x_vals_bfgs), x_vals_bfgs, label="Quasi-Newton (BFGS) method")
# Show the plot
plot(p, p2, layout=(1, 2))

x_damped_newton