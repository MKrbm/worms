using LinearAlgebra
using Plots
using ForwardDiff

# Define the objective function
function f(x)
    return (x[1] - 3)^2 + (x[2] - 2)^2 + abs(cos(x[1] * x[2])) * exp(max(-x[1], 3))
end

# Define the gradient using ForwardDiff
grad_f = x -> ForwardDiff.gradient(f, x)

# Define the Hessian using ForwardDiff
hess_f = x -> ForwardDiff.hessian(f, x)

# Newton's method for optimization
function newton_optimize(f, grad_f, hess_f, x0; tol=1e-6, max_iter=100)
    x = x0
    history = [x0]  # to store the path
    for i in 1:max_iter
        g = grad_f(x)
        H = hess_f(x)
        dx = -H \ g  # Newton's step
        x = x + dx
        push!(history, x)
        if norm(dx) < tol
            break
        end
    end
    return x, history
end

# Initial guess
x0 = [0.0, 0.0]

# Optimize
opt_x, history = newton_optimize(f, grad_f, hess_f, x0)

# Print the optimal solution
println("Optimal solution: ", opt_x)

# Extract the path of optimization
x_values = [x[1] for x in history]
y_values = [x[2] for x in history]

# Create a grid for contour plot
xgrid = range(-1, 6, length=300)
ygrid = range(-1, 5, length=200)
z = [f([x, y]) for y in ygrid, x in xgrid]

# Plot the contour and the path of optimization
p = plot()
contour(p, xgrid, ygrid, z, levels=20, linewidth=1.5, xlabel="x-axis", ylabel="y-axis")
plot!(x_values, y_values, seriestype=:path, marker=:circle, color=:red, label="Optimization Path")
scatter!(x_values, y_values, color=:red, label="Steps")
plot!(title="Newton's Method Optimization", xlabel="x-axis", ylabel="y-axis")
gui(p)
closeall()