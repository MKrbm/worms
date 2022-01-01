#include "../include/Shastry.hpp"

model::Shastry::Shastry(int Lx, int Ly, double J1, double J2, double h)
:Lx(Lx), Ly(Ly), J1(J1), J2(J2),
h(h), base_spin_model(return_lattice(Lx, Ly)){}
