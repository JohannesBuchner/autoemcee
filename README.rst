=========
autoemcee
=========

Runs MCMC automatically to convergence.

About
-----

Runs a family of Markov Chain Monte Carlo ensemble samplers (Affine-Invariant or Slice Sampler)
with gradually increasing number of samples until they converge.

Convergence is tested within each ensemble and across ensembles,
see `MCMC ensemble convergence test <mcmc-ensemble-convergence.html>`_
for details.

Supports parallelisation with MPI. No modifications to your code is needed, 
just run your script with mpiexec.

This package is built on top of emcee, zeus, anviz and mpi4py.

You can help by testing autoemcee and reporting issues. Code contributions are welcome.
See the `Contributing page <https://johannesbuchner.github.io/autoemcee/contributing.html>`_.

.. image:: https://img.shields.io/pypi/v/autoemcee.svg
        :target: https://pypi.python.org/pypi/autoemcee

.. image:: https://github.com/JohannesBuchner/autoemcee/actions/workflows/tests.yml/badge.svg
        :target: https://github.com/JohannesBuchner/autoemcee/actions/workflows/tests.yml

.. image:: https://coveralls.io/repos/github/JohannesBuchner/autoemcee/badge.svg?branch=master
        :target: https://coveralls.io/github/JohannesBuchner/autoemcee?branch=master

.. image:: https://img.shields.io/badge/docs-published-ok.svg
        :target: https://johannesbuchner.github.io/autoemcee/
        :alt: Documentation Status

Features
---------

* Pythonic. pip installable.
* Easy to program for: Sanity checks with meaningful errors
* both emcee and zeus are supported
* MPI support for parallel high-performance computing

Usage
^^^^^

Read the full documentation at:

https://johannesbuchner.github.io/autoemcee/


For parallelisation, use::

        mpiexec -np 4 python3 yourscript.py


Licence
^^^^^^^

GPLv3 (see LICENCE file). If you require another license, please contact me.

Icon made by `Vecteezy <https://www.flaticon.com/authors/smashicons>`_ from `Flaticon <https://www.flaticon.com/>`_ .


Other projects
^^^^^^^^^^^^^^

See also:

 * UltraNest: https://johannesbuchner.github.io/UltraNest/
 * snowline: https://johannesbuchner.github.io/snowline/
