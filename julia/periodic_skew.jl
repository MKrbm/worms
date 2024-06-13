using Pkg
Pkg.activate(".")
using LinearAlgebra
using RandomMatrix
using Plots


a = zeros((10, 10))

a[1, 4] = 1;
a[4, 1] = -1;
A = kron(a, I(size(a)[1]));

exp(2π * A)

# a = Symmetric(a);


# res = eigen(A);

# unique(res.values) 


