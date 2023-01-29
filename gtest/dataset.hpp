/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* Test set for accumulator testing
 */
#pragma once
#include <vector>
static const std::vector<double> heisenberg1D_ham_vector = {0.25, 0, 0, 0, 0, 0.75, 0.5, 0, 0, 0.5, 0.75, 0, 0, 0, 0, 0.25};
static const std::vector<std::vector<double>> heisenberg1D_ham = { { -0.25, 0, 0, 0}, { 0, 0.25, -0.5, 0}, { 0, -0.5, 0.25, 0}, { 0, 0, 0, -0.25} };