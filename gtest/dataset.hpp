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

static const std::vector<double> worm_obs_check = { 152.8 ,  83.42, 253.33,   7.95,  44.92,  81.57,  70.95, 176.72,
          126.09,  84.88,  68.77, 151.74, 204.77,  91.7 , 155.45, 133.93};


// Jz_-1_Jx_-0.5_Jy_-0.3_hz_0_hx_0.5
static const std::vector<double> check_warp_ham = {0.75,0.125,0.125,0.05,0.125,0.25,0.2,0.125,0.125,0.2,0.25,0.125,0.05,0.125,0.125, 0.75};

static const std::vector<bool> check_has_warp = {true,true,true,false,true,false,false,true,true,false,false,true,false,true,true,true};