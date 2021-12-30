#include "../include/Shastry.hpp"

model::Shastry::Shastry(int L, double J1, double J2, double h, int dim)
:J1(J1), J2(J2), dim(dim),
h(h), base_spin_model(lattice::graph::simple(dim, L)){}
