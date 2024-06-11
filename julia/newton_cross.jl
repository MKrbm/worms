using LinearAlgebra
using ForwardDiff
using ForwardDiff: gradient, hessian
using Zygote 
using Optim
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


function quasi_newton_bfgs(f, x0 :: Number; tol=1e-6, max_iter=100)
    x_vals :: Vector{Number} = []
    callback = (p) -> begin
        println(p.metadata)
        push!(x_vals, p.metadata["x"][1])
        false
    end
    options = Optim.Options(callback = callback, show_trace = false, extended_trace = true)
    result = optimize(f, [x0], BFGS(), options)
    # println(options)
    # println(result)
    return Optim.minimizer(result), Optim.iterations(result), x_vals
end

# Initial guess
x0 = -0.3  # Start with an initial guess close to the expected root

# Run the Newton method for root finding
# x_root, iter_root, x_vals_root = newton_root_method(g, x0)
# Run the Newton method for root finding
x_root_newton, iter_root_newton, x_vals_root_newton = newton_root_method(g, x0)

# Run the Quasi-Newton BFGS method for root finding
x_root_bfgs, iter_root_bfgs, x_vals_root_bfgs = quasi_newton_bfgs(g, x0)

# Print results
println("Newton's method root finding: x = $x_root_newton, iterations = $iter_root_newton")
println("Quasi-Newton (BFGS) method root finding: x = $x_root_bfgs, iterations = $iter_root_bfgs")

# Create a grid for the landscape of the target function
xgrid = range(-3, 6, length=300)
z = [f(x) for x in xgrid]

# Create the animation for the Newton method root finding process
anim_newton = @animate for i in eachindex(x_vals_root_newton)
    p = plot(xgrid, z, linewidth=1.5, xlabel="x-axis", ylabel="y-axis", title="Function Landscape with Newton's Method Root Finding Path", legend=:topright)
    scatter!(p, [x_vals_root_newton[i]], [f(x_vals_root_newton[i])], label="Newton's method root finding", color=:red, markersize=3)
    hline!(p, [0.2], label="y = 0.2", linestyle=:dash)
end

# Save the animation as a GIF
gif(anim_newton, "newton_root_finding.gif", fps=2)

# Create the animation for the Quasi-Newton BFGS method root finding process
anim_bfgs = @animate for i in eachindex(x_vals_root_bfgs)
    p = plot(xgrid, z, linewidth=1.5, xlabel="x-axis", ylabel="y-axis", title="Function Landscape with Quasi-Newton (BFGS) Root Finding Path", legend=:topright)
    scatter!(p, [x_vals_root_bfgs[i]], [f(x_vals_root_bfgs[i])], label="Quasi-Newton (BFGS) method root finding", color=:blue, markersize=3)
    hline!(p, [0.2], label="y = 0.2", linestyle=:dash)
end

# Save the animation as a GIF
gif(anim_bfgs, "quasi_newton_bfgs_root_finding.gif", fps=2)
