=========
autoemcee
=========

Runs MCMC automatically to convergence.

About
-----

Runs a family of Affine-Invariant Markov Chain Monte Carlo samplers (emcee)
with gradually increasing number of samples until they converge.

Convergence is tested for each ensemble and for each parameter with the Geweke diagnostic.
Additionally, across ensembles, the Gelman-Rubin r-hat is used.

Can be run with MPI without modifying the code.

.. image:: https://img.shields.io/pypi/v/autoemcee.svg
        :target: https://pypi.python.org/pypi/autoemcee

.. image:: https://api.travis-ci.org/JohannesBuchner/autoemcee.svg?branch=master&status=started
        :target: https://travis-ci.org/github/JohannesBuchner/autoemcee

.. image:: https://img.shields.io/badge/docs-published-ok.svg
        :target: https://johannesbuchner.github.io/autoemcee/
        :alt: Documentation Status


This package is built on top of emcee, anviz and mpi4py.

You can help by testing autoemcee and reporting issues. Code contributions are welcome.
See the `Contributing page <https://johannesbuchner.github.io/autoemcee/contributing.html>`_.

Features
---------

* Pythonic. pip installable.
* Easy to program for: Sanity checks with meaningful errors
* Fast
* MPI support

Usage
^^^^^

Read the full documentation at:

https://johannesbuchner.github.io/autoemcee/


Licence
^^^^^^^

GPLv3 (see LICENCE file). If you require another license, please contact me.

Icon made by `Vecteezy <https://www.flaticon.com/authors/smashicons>`_ from `Flaticon <https://www.flaticon.com/>`_ .
