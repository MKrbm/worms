<!--toc:start-->

- [Settings](#settings)
- [EXACT CALCULATION](#exact-calculation)
  - [How to calculate the exact expectation value of quantum spin system?](#how-to-calculate-the-exact-expectation-value-of-quantum-spin-system)
    - [jax_solver](#jaxsolver)
- [Notes](#notes)
- [APPROXIMATE CALCULATION (QMC)](#approximate-calculation-qmc)
  - [Instructions for using main_MPI.cpp with `model.cfg`](#instructions-for-using-mainmpicpp-with-modelcfg)
    - [Prerequisites](#prerequisites)
    - [Steps](#steps)
    - [Parameters](#parameters)
- [Comparison of exact and approximate calculation](#comparison-of-exact-and-approximate-calculation)
  - [HXYZ](#hxyz)
  - [Ising](#ising)
  - [Study on inconcistency](#study-on-inconcistency)
- [jax_optimizer.py](#jaxoptimizerpy)
  - [Note](#note)
  - [Memos](#memos)
    - [QES : quasi energy loss](#qes-quasi-energy-loss)
    - [SMES : system minimum energy loss](#smes-system-minimum-energy-loss)
    - [MES : minimize energy loss](#mes-minimize-energy-loss)
- [Test effect of add-first method](#test-effect-of-add-first-method)
- [Summarize results with table formalt](#summarize-results-with-table-formalt)
  - [Table :](#table)
  - [Note](#note)
  - [Optimize with python](#optimize-with-python)
    - [Expected result](#expected-result)
  - [Calculate with MC.](#calculate-with-mc)
- [Debug / Idea](#debug-idea)
  - [Warm warp](#warm-warp)

<!--toc:end-->

# Settings

- Dokcer : v4.21.1
- Remote-Container : v0.251.0

# EXACT CALCULATION

## How to calculate the exact expectation value of quantum spin system?

### jax_solver

To calculate the exact expectation value of a quantum spin system, you need to
use exact diagonalization. The diagonalization is done with the jax library.

1. Navigate to the parent directory of `solver_jax.py`
2. Run `python solver_jax.py` with the following arguments:
   - `-m` or `--model`: The name of the model to be simulated. Possible choices
     are "KH" or "HXYZ".
   - `-Jz` or `--coupling_z`: The z coupling constant, default value is 1.
   - `-Jx` or `--coupling_x`: The x coupling constant.
   - `-Jy` or `--coupling_y`: The y coupling constant.
   - `-hx` or `--mag_x`: The x magnetic field, default value is 0.
   - `-hz` or `--mag_z`: The z magnetic field, default value is 0.
   - `-T` or `--temperature`: The temperature.
   - `-L1` or `--length1`: The length of one side (required).
   - `-L2` or `--length2`: The length of the other side.
   - `-u` or `--unitary_algorithm`: Algorithm to determine the local unitary
     matrix. The default is "original".
   - `-gs`: If set, only the ground state is calculated.

For example: `python solver_jax.py -m HXYZ -L1 6 -Jx -.5 -Jy 0.5 -T 1`

3. The script will run the exact diagonalization on the selected model with the
   given parameters and save the ground state, eigenvalues, and statistical data
   (such as energy per site and specific heat) to the `out` directory.

4. Results are stored in the following formats:

   - Ground state: `.npy` and `.csv`
   - Eigenvalues: `.npy` and `.csv`
   - Statistical data: `.csv`

5. Important statistics are "eigenvalues.csv", "statistics.csv"

# Notes

- Currently only the KH (Kagome Heisenberg) and HXYZ models are supported.

- Also, only 1D is available for the HXYZ model due to the dimension of the
  Hilbert space.

- Filepath migh be changed in the future.

# APPROXIMATE CALCULATION (QMC)

## Instructions for using main_MPI.cpp with `model.cfg`

### Prerequisites

1. Ensure that the MPI (Message Passing Interface) library is installed on your
   machine. MPI is a standardized and portable message-passing system designed
   by a group of researchers from academia and industry to function on a wide
   variety of parallel computers.
2. Make sure you have the `model.cfg` file at the appropriate location.

### Steps

1. Set the configuration parameters in `model.cfg` according to the system you
   want to study. This includes parameters for the Monte Carlo simulation
   (number of sweeps, temperature, etc.), model parameters (basis, cell,
   hamiltonian path, etc.), and the particular model you want to use (Kagome1,
   Kagome3, SS1, SS2, etc.).

   - The parameters in `mc_settings` can be set globally with `default` or for a
     specific model under `config`.
   - Each model has its own set of parameters, which includes information about
     the lattice, Hamiltonian, and degrees of freedom, among others.

2. Compile `main_MPI.cpp`. This usually involves a command similar to
   `mpic++ -o main_MPI main_MPI.cpp`. Please note that the command might differ
   based on your particular setup.

3. Run the compiled file with `mpirun`. The number of processes should be
   defined as per the user's requirement. For instance, to run the code with 4
   processes, the command would look like this: `mpirun -np 4 ./main_MPI`.
   Again, this may vary depending on your system setup.

### Parameters

1. file: A string representing the path to an XML file which probably contains
   the specifications for different lattice structures.

2. basis: A string describing the basis of the lattice. Here, it's a "triangular
   lattice".

3. cell: A string describing the type of cell in the lattice. For example,
   "kagome" or "anisotropic triangular".

4. ham_path: A string representing the path to a file containing the Hamiltonian
   of the system.

5. length: A list of integers representing the dimensions of the lattice. For
   instance, [2, 2] would represent a 2D lattice with dimensions 2x2.

6. types: A list of integers representing the types of particles in the system.
   The purpose of the numbers isn't specified, but they could possibly represent
   different types of spins, different types of atoms in a solid, etc.

7. params: A list of floating-point numbers representing parameters of the
   model. These could be quantities like the magnetic field in a spin model, or
   the interaction strength in a lattice model.

8. dofs: "Degrees of freedom", a list of integers. In physics, the degrees of
   freedom of a system is the number of parameters that determine the state of a
   physical system.

9. shift: A floating-point number that might represent a shift in the energy
   levels or another relevant quantity in the system.

10. repeat: A boolean (true/false) value that could control whether certain
    aspects of the simulation are repeated.

11. zero_worm: A boolean value. Given the context, it may be related to worm
    algorithms, which are a type of Monte Carlo method used to study quantum
    systems.

12. ns_unit: An integer representing the number of sites per unit cell in the
    lattice.

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

    | temperature | alpha = 0  | 0.5        | 0.9        | exact               |
    | ----------- | ---------- | ---------- | ---------- | ------------------- |
    | 0.5         | -0.341425  | -0.303497  | -0.257014  | -0.3412430671362625 |
    | 10          | -0.0189388 | -0.0162121 | -0.0140394 | -0.019197767945     |

- ## Ising

  - Memo

    - All python executable files are in `python/rmsKit`.
    - Local hamiltonian was generated by `python jax_optimize.py -m Ising1D`.
    - Exact calculation was done by `python solver_jax.py -m Ising1D -L 4`.

  - 1D L = 4 [J, H] = [1, 0]

    - Table of results

      | temperature | alpha = 0 | 0.5 | 0.9 | exact | | ----------- |
      --------------- | ---------------- | ----- | ----- | | 0.5 | -0.133976 |
      -0.0641208 | -0.011604 | -0.133976 |

  - 1D L = 2 [J, H] = [1, 0]

        | temperature | alpha = 0 | 0.5 | 0.9 | exact |
        | ----------- | --------------- | ---------------- | ----- | ----- |
        | 0.5         | -0.190043       | -0.11459        | -0.025182 |  -0.19052 |

    - Even there is inconsistency on energies for L = 2 (which is simplest
      case).
    - Next we're going to study the reason of this. 02/08/2023.

  - ## Study on inconcistency

    - **Reason** for the bug.

      - need to consider matrix element even when worm go through the nn site.
        - Previously, I only let worm go and change the state for 100 %.
        - But in reality, worm can go through the nn site with some probability
          otherwise rejected.

    - Tests
      - gtest/Ising1D
        - comparing the result with `python jax_optimize.py -m Ising1D -L 4`
        - MC result must be within 3 sigma.
      - gtest/HXX
        - comparing the result with
          `python jax_optimize.py -m HXX -L 4 -Jx 1 -Jy 1 -Jz 1 -hx 0 -hz 0`
        - test : `HXX1D_02a` test the case of alpha = 0.2

# jax_optimizer.py

- ## Note
  - In the latest research, in some model, only system energy loss was
    effective.
  -
- ## Memos

  - ### loss functions

    - #### QES : quasi energy loss
      In the case of Kagome Heisenberg model.
      - ??
    - #### SMEL : system minimum energy loss
      - Just calculate ground energy of the system.
      - There might be the computationally efficient way to calculate
        differentiation of the ground energy.
        - But for now, exact diagonalization is used.
    - #### SEL : system energy loss
      - minimize expectation value of system energy at certain temperature.
    - #### MES : minimize energy loss
      - Minimize minimum energy of local hamiltonian

  - ### Warm warp
    - idea
      - forbid bond operators having warm warp.
        - indeed, warm warp only emerges when operator has single flip.
        - In this sense, the effect of warp can be interpreted into single flip.
        - Therefore, if we enable single-flip operator, warm warp among bond
          operator will be disabled.

# Test effect of add-first method

- HXYZ model on 2D lattice (commit 86fbd568d695de1dbc824b8102190c43d7535fa6)

  - generate local hamiltonian with
    `python jax_optimize.py -m HXYZ2D -loss none -u original -Jz 1 -Jx 1 -Jy 1`
  - calculate exact energy per site
    `python solver_jax.py -m HXYZ2D -L1 3 -L2 3 -Jx 1 -Jy 1 -Jz 1`
  - J = [-1, -1, -1] sweeps = 5\*E5
    - with sign problem
    - for T = 1,
      - exact energy per site E = -0.2947239498963471
      - QMC result E = -0.296527 +- 0.00151379
  - J = [-0.3, -0.5, -1] sweeps = 5\*E5

    - with sign problem
    - for T = 1,
      - exact energy per site E = -0.2042471924200272
      - QMC result E = -0.204224 +- 0.000689048
    - for T = 0.5,
      - exact energy per site E = -0.4332697322625007
      - QMC result E = -0.433781 +- 0.000514781

  - # Summarize results with table format

    - J1 = [-1, -1, -1] J2 = [-0.3, -0.5, -1] J3 = [0.3, 0.5 , 0.8] J4 = J3 + hx
      = 0.3 / L = [3, 3]

      - ### Table :

        |              | J1 (T = 1)              | J2 (T = 1)               | J3 (T = 1)               | J3 (T = 0.5)             | J3 (alpha = 0.2) (4\*E6 sweeps) | J4 (T = 1.0 sweeps = 50 _ 5 _ 1E6) | J4 (T = 1.0 sweeps = 50 _ 5 _ 1E6, alpha = 0.3) | J4 (T = 0.5 sweeps = 50 _ 5 _ 1E6, alpha = 0.3) | J4 (T = 0.5, M=10, alpha = 0.3, sweeps = 60 _ 1E6 _ 12) |
        | ------------ | ----------------------- | ------------------------ | ------------------------ | ------------------------ | ------------------------------- | ---------------------------------- | ----------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------- |
        | exact        | -0.2947239498963471     | -0.2042471924200272      | -0.10720369505809212     | -0.16921390812569803     | -0.16921390812569803            | -0.1199219559789621                | -0.1199219559789621                             | -0.1837105838760283                             | -0.1837105838760283                                     |
        | QMC          | -0.296527 +- 0.00151379 | -0.204224 +- 0.000689048 | -0.106831 +- 0.000565658 | -0.168934 +- 0.000404437 | -0.169261 +- 0.000191556        | -0.119987 +- 6.59584e-05           | -0.120032 +- 0.000152568                        | -0.18378 +- 0.000146586                         | -0.183632 +- 0.000133653                                |
        | average sign | 1                       | 1                        | 0.96896 +- 0.000475423   | 0.7855 +- 0.00153094     | 0.88507 +- 0.000599312          | 0.880948 +- 0.000869874            |                                                 | 0.367401 +- 0.000474544                         | 0.367439 +- 0.000233516                                 |

    - J4 = [-0.3, 0.8 0.5] hx = 0.3

      - comparison of loss function.
      - L = [2,4]

        energy

        | loss    | exact                | none | mes                      | sel (2x4)                |
        | ------- | -------------------- | ---- | ------------------------ | ------------------------ |
        | T = 0.2 | -0.42002640825612453 |      | -0.420426 +- 0.00044748  | -0.420277 +- 0.000273169 |
        | T = 0.5 | -0.31877822461530514 |      | -0.318601 +- 0.00044282  |                          |
        | T = 1.0 | -0.18543629571195416 |      | -0.185768 +- 0.000442904 | -0.18593 +- 0.000557636  |

        average sign

        | loss    | none | mes                     | sel (2x4) |
        | ------- | ---- | ----------------------- | --------- |
        | T = 0.2 |      | 0.406658 +- 0.001194    | 1         |
        | T = 0.5 |      | 0.805008 +- 0.000756936 | 1         |
        | T = 1.0 |      | 0.971264 +- 0.000246244 | 1         |

      - L = [10, 10], 1E*5 * 10

        energy

        | loss    | none | mes                    | sel (2x4)                |
        | ------- | ---- | ---------------------- | ------------------------ |
        | T = 0.2 |      | -0.438488 +- 0.0197447 | -0.421475 +- 8.98918e-05 |
        | T = 0.5 |      | ?                      | ?                        |
        | T = 1.0 |      | ?                      | ?                        |

        average sign

        | loss    | none | mes                    | sel (2x4) |
        | ------- | ---- | ---------------------- | --------- |
        | T = 0.2 |      | 0.002468 +- 0.00111234 | 1         |
        | T = 0.5 |      | ?                      | 1         |
        | T = 1.0 |      | ?                      | 1         |

      - ### Note
        - zero worm is required to simulate J4.
        - with this commit ( 86fbd568d695de1dbc824b8102190c43d7535fa6 ), J4
          cannot be simulated precisely maybe because zero worm doesn't work
          well.
          - fix the bug in commit
          - I found the bug for this commit fix this with the next commit
            (87e387c1d3fc401bda0531519bcde52b219a672f).
          - J4 (alpha = 0.2) is simulated with commit
            b62ce3c91b63f3ef7fbcab3e856ca90c1657a0e4

  - ## Optimize with python
    - ### Expected result
  - ## Calculate with MC.

- 1D HXYZ model

  - J = [-0.3, 0.8, 0.5], hx = 0.3

    - exact energy per site E = -0.07947479512910453
    - ### Table L = 9, N = 1E6,

      Energy

      | loss    | exact                | none | mes                       | sel (L = 9) | sel (L = 3)              |
      | ------- | -------------------- | ---- | ------------------------- | ----------- | ------------------------ |
      | T = 0.2 | -0.212676069550371   | ?    | -0.212939 +- 0.000523401  | -0.212427   | -0.213036 +- 0.000499919 |
      | T = 0.5 | -0.14008828428566178 | ?    | ?                         | -0.140461   | ?                        |
      | T = 1.0 | -0.07947479512910453 | ?    | -0.0797412 +- 0.000316406 | -0.0796332  | ?                        |

      Average sign

      | loss    | none | mes                     | sel (L = 9)             | sel (L = 3)            |
      | ------- | ---- | ----------------------- | ----------------------- | ---------------------- |
      | T = 0.2 |      | 0.318234 +- 0.00149475  | 0.969366 +- 0.000411697 | 0.316162 +- 0.00145224 |
      | T = 0.5 |      | ?                       | 0.99983 +- 2.73328e-05  |                        |
      | T = 1.0 |      | 0.987138 +- 0.000205856 | 1                       |                        |

- Kagome Heisenberg model

  - J = [1, 1, 1] L = [2 x 2] sps = 8

    - ## Table

      Energy

      | loss    | exact                | none                     | mes                      | sel (L = 2x2) | sel ( alpha = 0.1) |
      | ------- | -------------------- | ------------------------ | ------------------------ | ------------- | ------------------ |
      | T = 1   | -0.2906764218942059  | -0.291417 +- 0.000969184 | -0.289639 +- 0.000968546 |               |                    |
      | T = 0.5 | -0.38218971185385137 | -0.3651 +- 0.00856272    | -0.380632 +- 0.00344475  |               |                    |
      | T = 0.2 | -0.4242839047074232  | -0.896963 +- 0.297377    |                          |               |                    |

      average sign

      | loss    | none                    | mes                     | sel (L = 2x2) |
      | ------- | ----------------------- | ----------------------- | ------------- |
      | T = 1   | 0.605596 +- 0.00183426  | 0.602502 +- 0.00155575  |               |
      | T = 0.5 | 0.056876 +- 0.0015803   | 0.0560778 +- 0.00060679 |               |
      | T = 0.2 | -0.000806 +- 0.00149467 | ?                       |               |

    - ## Bugs
      - for u = 3site, there is a bug when turn-on zero_worm and single_flip at
        the same time, since I thought only single_flip is able to have
        zero_worm when it is enabled. (commit
        5e0d860929e4283287c47aac82505b22528d59f4)
        - fixed with the commit 058be4353306cc2f8c844d65c363ea394dd709cf
        - Above didn't solve every bugs, Indeed, the new worm.cpp and the
          previous one should return the same result but for model : alpha =
          0.2, J4 HXYZ2D (if alpha = 0, returns the same result)
          - previous : e =
          - new : e = -0.185175 +- 0.000481603
      - Also, Kagome3 with signle_flip emboddies some bugs (although the above
        problem may be the main reason of the bug).

  - J = J4

    - it seems above parameter region doesn't improve the result.

- HXYZ1D 2site unit J4 to solve bugs (commit :
  def5e012700ee2b9ea2e7be632f3572f82cdf0a4 )

  - ## Table

  Previous commit sweeps = 50 \* 1E6

  | loss  | exact                | none                      | mes (alpha = 0)           | mes (alpha = 0.1, zw)     | mes (alpha = 0.8, zw)     | mes (alpha a = 0.8, zw=false) |
  | ----- | -------------------- | ------------------------- | ------------------------- | ------------------------- | ------------------------- | ----------------------------- |
  | T=1   | -0.07947518797058616 | -0.0797444 +- 0.000989955 | -0.0794962 +- 3.65553e-05 | -0.0767406 +- 9.22165e-05 | -0.0768608 +- 4.09506e-05 | ?                             |
  | T=0.5 | -0.14013342736830942 | ?                         | -0.140261 +- 0.000326007  | -0.131929 +- 2.89219e-05  | -0.131934 +- 1.51194e-05  | -0.131945 +- 7.10646e-05      |

  Fixed bug

  | loss  | exact                | none                      | mes (alpha = 0)           | mes (alpha = 0.1, zw)     | mes (alpha = 0.8, zw)     | mes (alpha a = 0.8, zw=false) |
  | ----- | -------------------- | ------------------------- | ------------------------- | ------------------------- | ------------------------- | ----------------------------- |
  | T=1   | -0.07947518797058616 | -0.0797444 +- 0.000989955 | -0.0794962 +- 3.65553e-05 | -0.0795685 +- 0.000246453 | -0.0793597 +- 0.000176153 | ?                             |
  | T=0.5 | -0.14013342736830942 | ?                         | -0.140261 +- 0.000326007  | -0.131929 +- 2.89219e-05  | -0.140235 +- 0.000298051  | -0.140203 +- 0.000172569      |

- FF_1D model (commit : 10a2a27336ffb000cff4d06e2e7588984e449c3b)

  - Generate 1D frustration-free model using peps technique :
    https://arxiv.org/abs/0707.2260
    - Actually the method was bit modified so that the local Hamiltonian can be
      frustration-free for any length of chain.
  - In order to generate local FF hamiltonian, run e.g. :
    `python torch_optimize_loc.py -m FF1D -loss mel -o LION -e 1000 -M 10000 -lr 0.005 -r 2`
    - here, 1D means lattice is 1D chain. Note that 2D has yet to be
      implemented.
  - When you look through the code, you can find there are some parameters to
    generate ff model.

    ```python
    p = dict(
        sps=3,
        rank=2,
        dimension=d,
        us=1,
        seed=1,
    )
    ```

    - sps : degree of spin freedom per site. This will be determined by `-r`
      flag.
    - rank : rank of tensor network, need to be lower than sps in order to be ff
    - dimension : currently only 1D is available
    - us : number of sites per unit cell. Usually 1.
    - seed : random seed. The same seed generates the same local hamiltonian.

  - run_simulation.py

  - seed = 512

    - Use MPS generated randomly from normal distribution (unlike previously)
    - sweeps : 6 \* 5E6
    - loss = 0.00921

      - Theoretically, the gs energy for absolute system is lower than loss \* L

    - L = 6

      Energy

      | loss   | exact                | mes                       | none                      |
      | ------ | -------------------- | ------------------------- | ------------------------- |
      | T=1    | 0.14848181708495692  | 0.148396 +- 0.000117865   | 0.148396 +- 0.000117865   |
      | T=0.5  | 0.024505723254959377 | 0.0245954 +- 7.01752e-05  | 0.0245954 +- 7.01752e-05  |
      | T=0.25 | 0.004447369789880035 | 0.00437926 +- 4.15002e-05 | 0.00440814 +- 0.000424928 |
      | T=0.1  | 0.00383386671997593  | 0.004206 +- 0.00571919    | 0.004206 +- 0.00571919    |
      | T=0.01 | ?                    | 0.00383387 +- 0.000100001 | -0.142564 +- 0.160116     |

      Average Sign

      | loos     | mes                     | none                    |
      | -------- | ----------------------- | ----------------------- |
      | T=1      | 0.999934 +- 4.28854e-06 | 0.98369 +- 0.000202372  |
      | T=0.5    | 0.999694 +- 1.10713e-05 | 0.896842 +- 0.00064242  |
      | T=0.25   | 0.998954 +- 3.06621e-05 | 0.586869 +- 0.000703541 |
      | T=0.1    | 0.993936 +- 0.000410075 | 0.036592 +- 0.00165042  |
      | T = 0.01 | 0.974147 +- 0.00164656  | 0.001125 +- 0.00129898  |

    - L = 20

      Energy

      | loss     | mes                       |
      | -------- | ------------------------- |
      | T = 0.05 | 0.00342205 +- 3.24912e-05 |

      Average Sign

      | loss     | mes                   |
      | -------- | --------------------- |
      | T = 0.05 | 0.918315 +- 0.0032797 |

    - L = 100

      Energy

      | loss     | mes                       |
      | -------- | ------------------------- |
      | T = 0.05 | 0.00343649 +- 1.72491e-05 |

      Average Sign

      | loss     | mes                    |
      | -------- | ---------------------- |
      | T = 0.05 | 0.658012 +- 0.00474073 |

  - seed = 0

    - loss = 0.17387

    - L = 6

      Energy

      | T    | exact               | mes                      | none                  |
      | ---- | ------------------- | ------------------------ | --------------------- |
      | 1    | 0.25118684103027006 | 0.251156 +- 0.000311873  | 0.250061 +- 0.0011267 |
      | 0.5  | 0.10662484818538966 | 0.106549 +- 0.000278406  | 0.10926 +- 0.00700185 |
      | 0.25 | 0.04332188189002812 | 0.0441297 +- 0.000682354 | -0.421567 +- 0.162081 |
      | 0.1  | 0.01599607518647706 | -0.275572 +- 0.252839    | ?                     |

      Average Sign

      | T    | mes                        | none                      |
      | ---- | -------------------------- | ------------------------- |
      | 1    | 0.974711 +- 0.000225541    | 0.672731 +- 0.00136263    |
      | 0.5  | 0.818884 +- 0.000745897    | 0.0647213 +- 0.00096075   |
      | 0.25 | 0.272267 +- 0.00117836     | 0.00106233 +- 0.000761337 |
      | 0.1  | 0.000597333 +- 0.000811593 | ?                         |

Average sign = 0.974711 +- 0.000225541 Energy per site = 0.251156 +- 0.000311873
