# MONTE-CARLO METHODS FOR THE NEUTRON TRANSPORT EQUATION

This folder contains code for the paper "Monte-Carlo Methods for the
Neutron Transport Equation", by A.M.G. Cox, S.C. Harris,
A.E. Kyprianou and M. Wang.

The code is made available under the MIT license. See License.txt for
information.

The code covers simulations in 1D and 2D cases, and we summarise the
main files here:

## One-dimensional Slab Reactor

The 1D case considers the example described in Section 9.2-3. The files
associated with this are:

- *FixedPt.py*:
Implements code for computing the theoretical eigenvalue and
eigenfunction.

- *NTE1D.py*:
Implements Monte Carlo Methods for the 1D NTE, branching RW, many-to-one
and h-Transformed. 

- *1DPlots.py*:
Script to produce the output for the paper.

## Two-dimensional Reactor

The 2D case looks at a simple 2D model of a reactor. The numerical results
are discussed in Section 9.4 of the paper. Note that the code will not
currently replicate the output from the paper. To do this correctly, the
parameter `test_scale` in *NTE2D.py* should be changed to 10.

- *hTransf.py*:
Implements the h-Transfrom for the 2DNTE case.

- *NTE2D.py*:
Implements Monte Carlo Methods for the 2D NTE, branching and
many-to-one cases.

- *2DPlots.py*, *2DPlots2.py*:
Produces some of the plots for the paper in the case of
particle-filtered examples and comparing the vanilla cases
(without particle filters). Also functionality to try to plot the
eigenfunction of the NTE using MC methods (not
used in the paper).


## Auxiliary Code

- *ParticleFilter.py*
Implements generic useful particle filter tools and processes.

- *simulate_tools.py*:
Useful tools for plotting output of simulations.

- *Logger.py*:
Tools for saving text output of code.
