using LinearAlgebra
using ForwardDiff
using ForwardDiff: gradient, hessian
using SpecialFunctions
using Plots

# Define the 1D objective function
function f(x)
    return abs(cos(x)) * lgamma(x) - cos(x) + 0.5
end

# Define the function g(x) = f(x) - 0.2 to find roots where f(x) = 0.2
function g(x)
    return f(x[1]) - 0.2
end

# Define gradient for the root-finding function g
grad_g = x -> ForwardDiff.gradient(g, x)

# Newton's method for root finding
function newton_root_method(g, x0; tol=1e-6, max_iter=100)
    x = x0
    x_vals = [x0]
    for i in 1:max_iter
        g_val = g([x])
        grad_val = grad_g([x])[1]
        if abs(grad_val) < tol
            println("Gradient too small, method may not converge.")
            return x, i, x_vals
        end
        step = -g_val / grad_val
        x = x + step
        push!(x_vals, x)
        if abs(g_val) < tol
            return x, i, x_vals
        end
    end
    return x, max_iter, x_vals
end

# Initial guess
x0 = -0.3  # Start with an initial guess close to the expected root

# Run the Newton method for root finding
x_root, iter_root, x_vals_root = newton_root_method(g, x0)

# Print results
println("Newton's method root finding: x = $x_root, iterations = $iter_root")

# Create a grid for the landscape of the target function
xgrid = range(-3, 6, length=300)
z = [f(x) for x in xgrid]

# Create the animation for the root finding process
anim = @animate for i in eachindex(x_vals_root)
    p = plot(xgrid, z, linewidth=1.5, xlabel="x-axis", ylabel="y-axis", title="Function Landscape with Root Finding Path", legend=:topright)
    scatter!(p, [x_vals_root[i]], [f(x_vals_root[i])], label="Newton's method root finding", color=:red, markersize=3)
    hline!(p, [0.2], label="y = 0", linestyle=:dash)
    frame(anim, p)
end

# Save the animation as a GIF
gif(anim, "newton_root_finding.gif", fps=2)
gif(anim, "newton_root_finding.gif", fps=2)