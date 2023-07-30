# Settings

- Dokcer : v4.21.1
- Remote-Container : v0.251.0

# EXACT CALCULATION

## How to calculate the exact expectation value of quantum spin system?

### jax_solver

To calculate the exact expectation value of a quantum spin system, you need to use exact diagonalization. The diagonalization is done with the jax library.

1. Navigate to the parent directory of `solver_jax.py`
2. Run `python solver_jax.py` with the following arguments:
   - `-m` or `--model`: The name of the model to be simulated. Possible choices are "KH" or "HXYZ".
   - `-Jz` or `--coupling_z`: The z coupling constant, default value is 1.
   - `-Jx` or `--coupling_x`: The x coupling constant.
   - `-Jy` or `--coupling_y`: The y coupling constant.
   - `-hx` or `--mag_x`: The x magnetic field, default value is 0.
   - `-hz` or `--mag_z`: The z magnetic field, default value is 0.
   - `-T` or `--temperature`: The temperature.
   - `-L1` or `--length1`: The length of one side (required).
   - `-L2` or `--length2`: The length of the other side.
   - `-u` or `--unitary_algorithm`: Algorithm to determine the local unitary matrix. The default is "original".
   - `-gs`: If set, only the ground state is calculated.

For example: `python solver_jax.py -m HXYZ -L1 6 -Jx -.5 -Jy 0.5 -T 1`

3. The script will run the exact diagonalization on the selected model with the given parameters and save the ground state, eigenvalues, and statistical data (such as energy per site and specific heat) to the `out` directory.

4. Results are stored in the following formats:

   - Ground state: `.npy` and `.csv`
   - Eigenvalues: `.npy` and `.csv`
   - Statistical data: `.csv`

5. Important statistics are "eigenvalues.csv", "statistics.csv"

# Notes

- Currently only the KH (Kagome Heisenberg) and HXYZ models are supported.

- Also, only 1D is available for the HXYZ model due to the dimension of the Hilbert space.

- Filepath migh be changed in the future.

# APPROXIMATE CALCULATION (QMC)

## Instructions for using main_MPI.cpp with `model.cfg`

### Prerequisites

1. Ensure that the MPI (Message Passing Interface) library is installed on your machine. MPI is a standardized and portable message-passing system designed by a group of researchers from academia and industry to function on a wide variety of parallel computers.
2. Make sure you have the `model.cfg` file at the appropriate location.

### Steps

1. Set the configuration parameters in `model.cfg` according to the system you want to study. This includes parameters for the Monte Carlo simulation (number of sweeps, temperature, etc.), model parameters (basis, cell, hamiltonian path, etc.), and the particular model you want to use (Kagome1, Kagome3, SS1, SS2, etc.).

   - The parameters in `mc_settings` can be set globally with `default` or for a specific model under `config`.
   - Each model has its own set of parameters, which includes information about the lattice, Hamiltonian, and degrees of freedom, among others.

2. Compile `main_MPI.cpp`. This usually involves a command similar to `mpic++ -o main_MPI main_MPI.cpp`. Please note that the command might differ based on your particular setup.

3. Run the compiled file with `mpirun`. The number of processes should be defined as per the user's requirement. For instance, to run the code with 4 processes, the command would look like this: `mpirun -np 4 ./main_MPI`. Again, this may vary depending on your system setup.

### Parameters

1. file: A string representing the path to an XML file which probably contains the specifications for different lattice structures.

2. basis: A string describing the basis of the lattice. Here, it's a "triangular lattice".

3. cell: A string describing the type of cell in the lattice. For example, "kagome" or "anisotropic triangular".

4. ham_path: A string representing the path to a file containing the Hamiltonian of the system.

5. length: A list of integers representing the dimensions of the lattice. For instance, [2, 2] would represent a 2D lattice with dimensions 2x2.

6. types: A list of integers representing the types of particles in the system. The purpose of the numbers isn't specified, but they could possibly represent different types of spins, different types of atoms in a solid, etc.

7. params: A list of floating-point numbers representing parameters of the model. These could be quantities like the magnetic field in a spin model, or the interaction strength in a lattice model.

8. dofs: "Degrees of freedom", a list of integers. In physics, the degrees of freedom of a system is the number of parameters that determine the state of a physical system.

9. shift: A floating-point number that might represent a shift in the energy levels or another relevant quantity in the system.

10. repeat: A boolean (true/false) value that could control whether certain aspects of the simulation are repeated.

11. zero_worm: A boolean value. Given the context, it may be related to worm algorithms, which are a type of Monte Carlo method used to study quantum systems.

12. ns_unit: An integer representing the number of sites per unit cell in the lattice.

# Comparison of exact and approximate calculation

- ## HXYZ

  - 1D L = 11 J = [1, 1, 1] (with sign problem) T = 0.5

    - QMC (alpha = 0)

          therms(each process)    : 100000
          sweeps(each process)    : 500000
          sweeps(in total)        : 3000000
          ----------------------------------------

          beta                 = 2
          Total Energy         = -3.75568 +- 0.00291866
          Average sign         = 0.999395 +- 3.24474e-05
          Energy per site      = -0.341425 +- 0.000265333
          Specific heat        = 0.347901 +- 0.00208308

      - config is :

            file   "../config/lattices.xml";
            basis = "chain lattice";
            cell = "simple1d";
            ham_path = "****/HXYZ/original/none/Jx_1_Jy_1_Jz_1_hx_0_hz_0/H";
            worm_obs_path = ["../gtest/model_array/worm_obs/g_test5"];
            length = [6]; # shape of model.
            types = [0];
            params = [1.0];
            dofs = [2]; #defgree of freedom.
            shift = 0.1;
            repeat = false;
            zero_worm = false;
            alpha = 0;
  
  - QMC (alpha = 0.5)
    
        Elapsed time         = 5.508(5.164) sec
        Speed                = 108932 MCS/sec
        beta                 = 2
        Total Energy         = -3.33847 +- 0.0037028
        Average sign         = 0.999559 +- 8.5122e-05
        Energy per site      = -0.303497 +- 0.000336619
        Specific heat        = 0.483836 +- 0.0025603

    - same config above

  - EXACT

    - enegy per site = `-0.3412430671362625`,


  - There is a bug. If alpha is not 0, the result is not correct.

  - Table of results

    | temperature | alpha = 0 | 0.5 | 0.9 | exact |
    | ----------- | --------------- | ---------------- | ----- | ----- |
    | 0.5         | -0.341425       | -0.303497        | -0.257014 |  -0.3412430671362625 |
    | 10          | -0.0189388       | -0.0162121        | -0.0140394 |  -0.019197767945 |