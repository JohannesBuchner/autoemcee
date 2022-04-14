=========
autoemcee
=========

Runs MCMC automatically to convergence.

About
-----

Runs a family of Markov Chain Monte Carlo ensemble samplers (Affine-Invariant or Slice Sampler)
with gradually increasing number of samples until they converge.

Convergence is tested for each ensemble and for each parameter with the Geweke diagnostic.
Additionally, across ensembles, the Gelman-Rubin r-hat is used.

Supports MPI, without requiring any modifications to your code (just run your script with mpiexec).

.. image:: https://img.shields.io/pypi/v/autoemcee.svg
        :target: https://pypi.python.org/pypi/autoemcee

.. image:: https://api.travis-ci.com/JohannesBuchner/autoemcee.svg?branch=master&status=started
        :target: https://travis-ci.com/github/JohannesBuchner/autoemcee

.. image:: https://img.shields.io/badge/docs-published-ok.svg
        :target: https://johannesbuchner.github.io/autoemcee/
        :alt: Documentation Status

This package is built on top of emcee, zeus, anviz and mpi4py.

You can help by testing autoemcee and reporting issues. Code contributions are welcome.
See the `Contributing page <https://johannesbuchner.github.io/autoemcee/contributing.html>`_.

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
