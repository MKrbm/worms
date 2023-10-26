/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* Test set for accumulator testing
 */
#pragma once
#include <vector>
static const std::vector<double> heisenberg1D_ham_vector = {
    0.25, 0, 0, 0, 0, 0.75, 0.5, 0, 0, 0.5, 0.75, 0, 0, 0, 0, 0.25};
static const std::vector<std::vector<double>> heisenberg1D_ham = {
    {-0.25, 0, 0, 0}, {0, 0.25, -0.5, 0}, {0, -0.5, 0.25, 0}, {0, 0, 0, -0.25}};

static const std::vector<double> worm_obs_check = {
    152.8,  83.42, 253.33, 7.95,   44.92,  81.57, 70.95,  176.72,
    126.09, 84.88, 68.77,  151.74, 204.77, 91.7,  155.45, 133.93};

// Jz_-1_Jx_-0.5_Jy_-0.3_hz_0_hx_0.5
static const std::vector<double> check_warp_ham = {
    0.75,  0.125, 0.125, 0.05,  0.125, 0.25,  0.2,   0.125,
    0.125, 0.2,   0.25,  0.125, 0.05,  0.125, 0.125, 0.75};

static const std::vector<bool> check_has_warp = {
    true, true,  true,  false, true,  false, false, true,
    true, false, false, true,  false, true,  true,  true};

// kagome bonds
static const std::vector<std::vector<int>> kagome_bonds = {
    {0, 1}, {0, 2}, {1, 2},  {0, 4},  {1, 11},  {0, 8}, {3, 4},  {3, 5},
    {4, 5}, {3, 1}, {4, 8},  {3, 11}, {6, 7},   {6, 8}, {7, 8},  {6, 10},
    {7, 5}, {6, 2}, {9, 10}, {9, 11}, {10, 11}, {9, 7}, {10, 2}, {9, 5}};

// >>> h
// array([[-0.1999532 ,  0.07508575,  0.07508575,  0.19999997],
//        [ 0.07508575,  0.19999997, -0.05000003,  0.07491425],
//        [ 0.07508575, -0.05000003,  0.19999997,  0.07491425],
//        [ 0.19999997,  0.07491425,  0.07491425, -0.20004675]])
// >>> u
// array([[ 1.55911151e-04, -9.99999988e-01],
//        [ 9.99999988e-01,  1.55911151e-04]])

static const std::vector<std::vector<double>> HXYZ1D_ham_u = {
    {-0.1999532, 0.07508575, 0.07508575, 0.19999997},
    {0.07508575, 0.19999997, -0.05000003, 0.07491425},
    {0.07508575, -0.05000003, 0.19999997, 0.07491425},
    {0.19999997, 0.07491425, 0.07491425, -0.20004675}
};

static const std::vector<std::vector<double>> HXYZ1D_unitary = {
    {1.55911151e-04, -9.99999988e-01}, {9.99999988e-01, 1.55911151e-04}
};
