"""Calculates the Bayesian evidence and posterior samples of arbitrary monomodal models."""

from __future__ import print_function
from __future__ import division

import os
import sys
import logging
import warnings

import numpy as np

import emcee
import arviz

__all__ = ['ReactiveAffineInvariantSampler']
__author__ = """Johannes Buchner"""
__email__ = 'johannes.buchner.acad@gmx.com'
__version__ = '0.1.0'


# Some parts are from the nnest library by Adam Moss (https://github.com/adammoss/nnest)
def create_logger(module_name, log_dir=None, level=logging.DEBUG):
    """
    Set up the logging channel `module_name`.

    Append to ``debug.log`` in `log_dir` (if not ``None``).
    Write to stdout with output level `level`.

    If logging handlers are already registered, no new handlers are
    registered.
    """
    logger = logging.getLogger(str(module_name))
    logger.setLevel(level)
    first_logger = logger.handlers == []
    if log_dir is not None and first_logger:
        # create file handler which logs even debug messages
        handler = logging.FileHandler(os.path.join(log_dir, 'debug.log'))
        formatter = logging.Formatter(
            '%(asctime)s [{}] [%(levelname)s] %(message)s'.format(module_name),
            datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    if first_logger:
        # if it is new, register to write to stdout
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter('[{}] %(message)s'.format(module_name))
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def vectorize(function):
    """Vectorize likelihood or prior_transform function."""
    def vectorized(args):
        """ vectorized version of function"""
        return np.asarray([function(arg) for arg in args])

    vectorized.__name__ = function.__name__
    return vectorized


class ReactiveAffineInvariantSampler(object):
    """Emcee sampler with reactive exploration strategy."""

    def __init__(self,
                 param_names,
                 loglike,
                 transform=None,
                 num_test_samples=2,
                 vectorized=False,
                 ):
        """Initialise Affine-Invariant sampler.

        Parameters
        -----------
        param_names: list of str, names of the parameters.
            Length gives dimensionality of the sampling problem.

        loglike: function
            log-likelihood function.
            Receives multiple parameter vectors, returns vector of likelihood.
        transform: function
            parameter transform from unit cube to physical parameters.
            Receives multiple cube vectors, returns multiple parameter vectors.

        num_test_samples: int
            test transform and likelihood with this number of
            random points for errors first. Useful to catch bugs.
        """
        self.paramnames = param_names
        x_dim = len(self.paramnames)

        self.sampler = 'reactive-importance'
        self.x_dim = x_dim
        self.log = True

        if self.log:
            self.logger = create_logger('autoemcee')

        if not vectorized:
            loglike = vectorize(loglike)
            if transform is not None:
                transform = vectorize(transform)

        self.ncall = 0
        self._set_likelihood_function(transform, loglike, num_test_samples)

    def _set_likelihood_function(self, transform, loglike, num_test_samples, make_safe=False):
        """Store the transform and log-likelihood functions.

        Tests with `num_test_samples` whether they work and give the correct output.

        if make_safe is set, make functions safer by accepting misformed
        return shapes and non-finite likelihood values.
        """
        # do some checks on the likelihood function
        # this makes debugging easier by failing early with meaningful errors
        
        # test with num_test_samples random points
        u = np.random.uniform(size=(num_test_samples, self.x_dim))
        p = transform(u) if transform is not None else u
        assert p.shape == (num_test_samples, self.x_dim,), ("Error in transform function: returned shape is %s, expected %s" % (p.shape, self.x_dim))
        logl = loglike(p)
        assert np.logical_and(u > 0, u < 1).all(), ("Error in transform function: u was modified!")
        assert logl.shape == (num_test_samples,), ("Error in loglikelihood function: returned shape is %s, expected %s" % (logl.shape, num_test_samples))
        assert np.isfinite(logl).all(), ("Error in loglikelihood function: returned non-finite number: %s for input u=%s p=%s" % (logl, u, p))

        self.loglike = loglike
        self.ncall = 0

        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform

    def _emcee_logprob(self, u):
        mask = np.logical_and((u > 0).all(axis=1), (u < 1).all(axis=1))
        L = -np.inf * np.ones(len(u))

        p = self.transform(u[mask, :])
        L[mask] = self.loglike(p)
        return L

    def find_starting_walkers(self, num_global_samples, num_walkers):
        assert num_global_samples > num_walkers, (num_global_samples, num_walkers)

        ndim, loglike, transform = self.x_dim, self.loglike, self.transform

        if self.log:
            self.logger.debug("global sampling for starting point ...")

        u = np.random.uniform(size=(num_global_samples, ndim))
        p = transform(u)
        L = loglike(p)
        
        # find indices of the highest likelihood ones
        i = np.argsort(L)[::-1][:num_walkers]
        
        return u[i, :], p[i, :], L[i]


    def run(self,
            num_global_samples=100,
            num_chains=4,
            num_walkers=None,
            max_ncalls=1000000,
            min_ess=400,
            max_improvement_loops=4,
            num_initial_steps=100,
            min_autocorr_times=50,
            verbose=True):
        """Sample at least *min_ess* effective samples have been drawn.

        The steps are:

        1. Draw *num_global_samples* from prior. The highest *num_walkers* points are selected.
        2. Optimize to find maximum likelihood point.
        3. Estimate local covariance with finite differences.
        4. Importance sample from Laplace approximation (with *num_gauss_samples*).
        5. Construct Gaussian mixture model from samples
        6. Simplify Gaussian mixture model with Variational Bayes
        7. Importance sample from mixture model

        Steps 5-7 are repeated (*max_improvement_loops* times).
        Steps 2-3 are performed with MINUIT, Steps 3-6
        are performed with pypmc.

        Parameters
        ----------
        min_ess: float
            Number of effective samples to draw
        max_ncalls: int
            Maximum number of likelihood function evaluations
        max_improvement_loops: int
            Number of times the proposal should be improved

        num_gauss_samples: int
            Number of samples to draw from initial Gaussian likelihood approximation before
            improving the approximation.
        num_global_samples: int
            Number of samples to draw from the prior to
        """
        
        if num_walkers is None:
            num_walkers = min(100, 2 * self.x_dim)

        num_steps = num_initial_steps
        self.logger.info("finding starting points and running initial %d MCMC steps" % (num_steps))
        self.samplers = []
        for chain in range(num_chains):
            u, p, L = self.find_starting_walkers(num_global_samples, num_walkers)
            
            sampler = emcee.EnsembleSampler(num_walkers, self.x_dim, self._emcee_logprob, vectorize=True)
            self.samplers.append(sampler)
            sampler.run_mcmc(u, num_steps)
        self.ncall += num_chains * (num_global_samples + (num_steps + 1) * num_walkers)
        
        for it in range(max_improvement_loops):
            self.logger.debug("checking convergence (iteration %d) ..." % (it+1))
            converged = True
            # check state of chains:
            for sampler in self.samplers:
                chain = sampler.get_chain()
                assert chain.shape == (num_steps, num_walkers, self.x_dim)
                accepts = (chain[1:, :, :] != chain[:-1, :, :]).any(axis=2).sum(axis=0)
                assert accepts.shape == (num_walkers,)
                print("accepts:", accepts * 100. / num_steps)
                
                # flatchain = sampler.get_chain(flat=True)
                
                # diagnose this chain
                
                # 0. analyse each variable
                max_autocorrlength = 1
                for i in range(self.x_dim):
                    chain_variable = chain[:, :, i]
                    """
                    # 1. treat each walker as a independent chain
                    try:
                        for w in range(num_walkers):
                            chain_walker = chain_variable[:, w]
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                tau = emcee.autocorr.integrated_time(chain_walker, c=5, tol=50, quiet=False)
                                tau = max(tau, 1)
                            max_autocorrlength = max(max_autocorrlength, tau)
                            if len(chain_walker) / tau < min_autocorr_times:
                                self.logger.debug("autocorrelation is long for parameter '%s': tau=%.1f -> %dx lengths" % (self.paramnames[i], tau, num_steps / tau))
                                converged = False
                                break
                    except emcee.autocorr.AutocorrError:
                        max_autocorrlength = np.inf
                        self.logger.debug("autocorrelation is too long for parameter '%s' to be estimated" % (self.paramnames[i]))
                        converged = False
                        break

                    if not converged:
                        break
                    """

                    # secondly, detect drift with geweke
                    a = sampler.get_chain(flat=True)[:num_steps // 4, i]
                    b = sampler.get_chain(flat=True)[-num_steps // 4:, i]
                    geweke_z = (a.mean() - b.mean()) / (np.var(a) + np.var(b))**0.5
                    if geweke_z > 2.:
                        self.logger.debug("geweke drift (z=%.1f) detected for parameter '%s'" % (geweke_z, self.paramnames[i]))
                        converged = False
                    
                #self.logger.debug("autocorrelation length: tau=%.1f -> %dx lengths" % (max_autocorrlength, num_steps / max_autocorrlength))
                if not converged:
                    break
                
            if converged:
                # finally, gelman-rubin diagnostic on chains
                chains = arviz.convert_to_dataset(np.asarray([sampler.get_chain(flat=True) for sampler in self.samplers]))
                rhat = arviz.rhat(chains).x.data
                self.logger.debug("rhat: %s (<1.2 is good)", rhat)
                converged = np.all(rhat < 1.2)
            
            if converged:
                self.logger.info("converged!!!")
                break
            
            self.logger.info("not converged yet at iteration %d" % (it+1))
            #self.logger.error("error at iteration %d" % (it+1))
            last_num_steps = num_steps
            num_steps = int(last_num_steps * 4)
            
            self.ncall += num_chains * (num_steps + 1 + last_num_steps) * num_walkers
            self.logger.info("Running %d MCMC steps ..." % (num_steps))
            for sampler in self.samplers:
                chain = sampler.get_chain(flat=True)
                # get a scale small compared to the width of the current posterior
                std = np.clip(chain.std(axis=0) * 1e-5, a_min=1e-30, a_max=1)
                assert std.shape == (self.x_dim,), std.shape
                
                # sample num_walkers points from chain, proportional to likelihood
                L = sampler.get_log_prob(flat=True)
                p = np.exp(L - L.max())
                i = np.random.choice(len(L), size=num_walkers, p=p / p.sum())
                # select points
                u = np.clip(chain[i, :], 1e-10, 1 - 1e-10)
                # add a bit of noise
                noise = np.random.normal(0, std, size=(num_walkers, self.x_dim))
                u = u + noise
                # avoid border
                u = np.clip(u, 1e-10, 1 - 1e-10)
                assert u.shape == (num_walkers, self.x_dim), (u.shape, (num_walkers, self.x_dim))
                
                self.logger.info("Starting at %s +- %s", u.mean(axis=0), u.std(axis=0))
                sampler.reset()
                #self.logger.info("not converged yet at iteration %d" % (it+1))
                state = sampler.run_mcmc(u, last_num_steps)
                sampler.reset()
                sampler.run_mcmc(state, num_steps)
            

"""

    def _update_results(self, samples, weights):
        if self.log:
            self.logger.info('Likelihood function evaluations: %d', self.ncall)

        integral_estimator = weights.sum() / len(weights)
        integral_uncertainty_estimator = np.sqrt((weights**2).sum() / len(weights) - integral_estimator**2) / np.sqrt(len(weights) - 1)

        logZ = np.log(integral_estimator)
        logZerr = np.log(integral_estimator + integral_uncertainty_estimator) - logZ
        ess_fraction = ess(weights)

        # get a decent accuracy based on the weights, and not too few samples
        Nsamples = int(max(400, ess_fraction * len(weights) * 40))
        eqsamples_u = resample_equal(samples, weights / weights.sum(), N=Nsamples)
        eqsamples = np.asarray([self.transform(u) for u in eqsamples_u])

        results = dict(
            z=integral_estimator,
            zerr=integral_uncertainty_estimator,
            logz=logZ,
            logzerr=logZerr,
            ess=ess_fraction,
            paramnames=self.paramnames,
            ncall=int(self.ncall),
            posterior=dict(
                mean=eqsamples.mean(axis=0).tolist(),
                stdev=eqsamples.std(axis=0).tolist(),
                median=np.percentile(eqsamples, 50, axis=0).tolist(),
                errlo=np.percentile(eqsamples, 15.8655, axis=0).tolist(),
                errup=np.percentile(eqsamples, 84.1345, axis=0).tolist(),
            ),
            samples=eqsamples,
        )
        self.results = results

    def print_results(self):
        "" "Give summary of marginal likelihood and parameters."" "
        if self.log:
            print()
            print('logZ = %(logz).3f +- %(logzerr).3f' % self.results)
            print()
            for i, p in enumerate(self.paramnames):
                v = self.results['samples'][:, i]
                sigma = v.std()
                med = v.mean()
                if sigma == 0:
                    i = 3
                else:
                    i = max(0, int(-np.floor(np.log10(sigma))) + 1)
                fmt = '%%.%df' % i
                fmts = '\t'.join(['    %-20s' + fmt + " +- " + fmt])
                print(fmts % (p, med, sigma))
"""
