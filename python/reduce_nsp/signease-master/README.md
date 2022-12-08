# signease -- A python package to ease the Quantum Monte Carlo sign problem, v.1

signease is intended to be a python package that in a mature stage shall allow to compute efficient basis changes of local Hamiltonian models to improve the sampling complexity of quantum Monte-Carlo algorithms. 

The currently implemented functionality of the package is limited to 2-local Hamiltonian and identical on-site orthogonal circuits. It provides the entire 
code used for the numerical study in the publication 

* Hangleiter, Roth, Nagaj, and Eisert. *Easing the Monte Carlo sign problem*
	[https://arxiv.org/abs/1906.02309](https://arxiv.org/abs/1906.02309)


We intend to considerably extend the package in the future. 


## Requirements

The code is developed with Python 3.6.7. and uses the following fairly standard packages in some recent version: 
`numpy, scipy, matplotlib, copy, itertools, warnings, os, multiprocessing`.

## Demo 

A simple example use case can be found in `simple_plots_frustrated_Ladder.py`
Run and wait for some time ...
```
	python simple_plots_frustrated_Ladder.py
```
This should display some plots.

## Reproducing the plots of the publication 

Executable examples as presented in the publication are:
	- `produce_j0model_data.py`
	- `produce_randomStoquastic_data.py`
	- `produce_frustratedLadder_data.py`

These scripts use the wrapper
	`f_optimisation.py`
to access the functionality provided by the circuit optimiser module
and use parallelisation to produce the data in the `/data` directory 
for the plots in the publication. With standard desktop computing power 
theses script run for about two hours. Each call of the scripts will
use a different seed for the random generators causing variations in 
the resulting plots. 

Subsequently, running the corresponding plot sripts 
	`plotscript_j0model.py`
	`plotscript_randomStoquastic.py`
	`plotscript_frustratedLadder_full.py`
generates the plot in the `/plot` directory displayed in the publication. 

## Overview 
The core functionality is provided by the classes defined in the `circuitOptimizer` module. 

The most important class types it provides are the follwing:
* The class `Measure` implementing a couple of non-stoquasticity measures.
* The type `Hamiltonian` implements the blue print of a local Hamiltonian model. 
	Specific models should be implement as childrens of the `Hamiltonian` class. 
* The type `Circuit` specifies the blue print of a local quantum circuit consisting of multiple Orthogonal matrices.
	A specific circuit should implement as a subclass of `Circuit`.
* Instances of the `Optimizer` class can perform a conjugate gradient optimisation of 
	an instace of `Measure` for a given Hamiltonian and circuit model, specified by 
	instances of `Hamiltonian` and `Circuit`.




## Contributing

We invite contributions to our code base in order to improve the package and 
make it widely applicable to many Hamiltonian models. Please contact us
via mail. 


## Authors

**Dominik Hangleiter** and **Ingo Roth** 


## How to cite 

Over and above the legal restrictions imposed by the license, if you use this software for an academic publication then you are obliged to provide proper attribution to the paper that describes it 

* Hangleiter, Roth, Nagaj, and Eisert. *Easing the Monte Carlo sign problem*
	[https://arxiv.org/abs/1906.02309](https://arxiv.org/abs/1906.02309)

and this code directly 

* Hangleiter, Roth: ***signease** -- A python package to ease the Quantum Monte Carlo sign problem, v.1* (2019). [https://gitlab.com/ingo.roth/signease](https://gitlab.com/ingo.roth/signease).

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* The optimisation routine is based on the algorithm of 
	Abrudan, Traian, Jan Eriksson, and Visa Koivunen. 
	"Conjugate Gradient Algorithm for Optimization under Unitary Matrix Constraint." Signal Processing 89, no. 9 (September 2009): 1704–14. 
	[https://doi.org/10.1016/j.sigpro.2009.03.015](https://doi.org/10.1016/j.sigpro.2009.03.015).

* The implementation of the optimisation routine and the parallelisation is based on code generously provided by Christian Krumnow. 
