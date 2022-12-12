"""Calculates the Bayesian evidence and posterior samples of arbitrary monomodal models."""

from __future__ import print_function
from __future__ import division

import os
import sys
import logging
import warnings
import corner

import numpy as np

import emcee
import arviz

__all__ = ['ReactiveAffineInvariantSampler']
__author__ = """Johannes Buchner"""
__email__ = 'johannes.buchner.acad@gmx.com'
__version__ = '0.4.1'


# Some parts are from the nnest library by Adam Moss (https://github.com/adammoss/nnest)
def create_logger(module_name, log_dir=None, level=logging.INFO):
    """
    Set up the logging channel `module_name`.

    Append to ``debug.log`` in `log_dir` (if not ``None``).
    Write to stdout with output level `level`.

    If logging handlers are already registered, no new handlers are
    registered.
    """
    logger = logging.getLogger(str(module_name))
    logger.setLevel(logging.DEBUG)
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
        #logger.setLevel(level)
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
                 sampler='goodman-weare',
                 ):
        """Initialise sampler.

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
        sampler: str
            if 'goodman-weare': use Goodman & Weare's affine invariant MCMC ensemble sampler
            if 'slice': use Karamanis & Beutler (2020)'s ensemble slice sampler
        vectorized: bool
            if true, likelihood and transform receive arrays of points, and return arrays

        num_test_samples: int
            test transform and likelihood with this number of
            random points for errors first. Useful to catch bugs.
        """
        self.paramnames = param_names
        x_dim = len(self.paramnames)

        self.sampler = 'reactive-importance'
        self.x_dim = x_dim

        self.ncall = 0
        self.use_mpi = False
        self.sampler = sampler
        if sampler not in ('goodman-weare', 'slice'):
            raise ValueError("sampler needs to be one of ('goodman-weare', 'slice')")
        try:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.mpi_size = self.comm.Get_size()
            self.mpi_rank = self.comm.Get_rank()
            if self.mpi_size > 1:
                self.use_mpi = True
                self._setup_distributed_seeds()
        except Exception:
            self.mpi_size = 1
            self.mpi_rank = 0
        
        self.log = self.mpi_rank == 0
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

        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform

    def _setup_distributed_seeds(self):
        if not self.use_mpi:
            return
        seed = 0
        if self.mpi_rank == 0:
            seed = np.random.randint(0, 1000000)

        seed = self.comm.bcast(seed, root=0)
        if self.mpi_rank > 0:
            # from http://arxiv.org/abs/1005.4117
            seed = int(abs(((seed * 181) * ((self.mpi_rank - 83) * 359)) % 104729))
            # print('setting seed:', self.mpi_rank, seed)
            np.random.seed(seed)

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
            num_global_samples=10000,
            num_chains=4,
            num_walkers=None,
            max_ncalls=1000000,
            max_improvement_loops=4,
            num_initial_steps=100,
            min_autocorr_times=0,
            rhat_max=1.01,
            geweke_max=2.,
            progress=True):
        """Sample until MCMC chains have converged.

        The steps are:

        1. Draw *num_global_samples* from prior. The highest *num_walkers* points are selected.
        2. Set *num_steps* to *num_initial_steps*
        3. Run *num_chains* MCMC ensembles for *num_steps* steps
        4. For each walker chain, compute auto-correlation length (Convergence requires *num_steps*/autocorrelation length > *min_autocorr_times*)
        5. For each parameter, compute geweke convergence diagnostic (Convergence requires |z| < geweke_max)
        6. For each ensemble, compute gelman-rubin rank convergence diagnostic (Convergence requires rhat<rhat_max)
        7. If converged, stop and return results. 
        8. Increase *num_steps* by 10, and repeat from (3) up to *max_improvement_loops* times.


        Parameters
        ----------

        num_global_samples: int
            Number of samples to draw from the prior to
        num_chains: int
            Number of independent ensembles to run. If running with MPI,
            this is set to the number of MPI processes.
        num_walkers: int
            Ensemble size. If None, max(100, 4 * dim) is used
        max_ncalls: int
            Maximum number of likelihood function evaluations
        num_initial_steps: int
            Number of sampler steps to take in first iteration
        max_improvement_loops: int
            Number of times MCMC should be re-attempted (see above)
        min_autocorr_times: float
            if positive, additionally require for convergence that the 
            number of samples is larger than the *min_autocorr_times*
            times the autocorrelation length.
        geweke_max: float
            Maximum absolute z-score of the geweke test allowed for convergence.
        rhat_max: float
            Maximum r-hat allowed for convergence.
        progress: bool
            if True, show progress bars

        """
        
        if num_walkers is None:
            num_walkers = max(100, 4 * self.x_dim)

        num_steps = num_initial_steps
        if self.use_mpi:
            num_chains = self.mpi_size
            num_chains_here = 1
        else:
            num_chains_here = num_chains

        if self.log:
            self.logger.info("finding starting points and running initial %d MCMC steps" % (num_steps))
        
        self.ncall = 0
        ncall_here = 0
        self.samplers = []
        for chain in range(num_chains_here):
            u, p, L = self.find_starting_walkers(num_global_samples, num_walkers)
            ncall_here += num_global_samples
            
            if self.sampler == 'goodman-weare':
                sampler = emcee.EnsembleSampler(num_walkers, self.x_dim, self._emcee_logprob, vectorize=True)
            elif self.sampler == 'slice':
                import zeus
                sampler = zeus.EnsembleSampler(nwalkers=num_walkers, ndim=self.x_dim, logprob_fn=self._emcee_logprob, vectorize=True,
                    maxiter=1e10, maxsteps=1e10)
            self.samplers.append(sampler)
            sampler.run_mcmc(u, num_steps, progress=self.log and progress)
            ncall_here += num_walkers
            ncall_here += getattr(sampler, 'ncall', num_steps * num_walkers)

        if self.use_mpi:
            recv_ncall = self.comm.gather(ncall_here, root=0)
            ncall_here = sum(self.comm.bcast(recv_ncall, root=0))

        assert ncall_here > 0, ncall_here
        self.ncall += ncall_here

        for it in range(max_improvement_loops):
            if self.log:
                self.logger.debug("checking convergence (iteration %d) ..." % (it+1))
            converged = True
            # check state of chains:
            for sampler in self.samplers:
                chain = sampler.get_chain()
                assert chain.shape == (num_steps, num_walkers, self.x_dim), (chain.shape, (num_steps, num_walkers, self.x_dim))
                accepts = (chain[1:, :, :] != chain[:-1, :, :]).any(axis=2).sum(axis=0)
                assert accepts.shape == (num_walkers,)
                if self.log:
                    i = np.argsort(accepts)
                    self.logger.debug(
                        "acceptance rates: %s%% (worst few)", 
                        (accepts[i[:8]] * 100. / (num_steps - 1)).astype(int))
                flat_chain = sampler.get_chain(flat=True)
                
                # diagnose this chain
                
                # 0. analyse each variable
                max_autocorrlength = 1
                for i in range(self.x_dim):
                    chain_variable = chain[:, :, i]
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
                        if min_autocorr_times > 0:
                            self.logger.debug("autocorrelation is too long for parameter '%s' to be estimated" % (self.paramnames[i]))
                            converged = False
                            break

                    if not converged:
                        break

                    # secondly, detect drift with geweke
                    a = flat_chain[:len(flat_chain) // 4, i]
                    b = flat_chain[-len(flat_chain) // 4:, i]
                    geweke_z = (a.mean() - b.mean()) / (np.var(a) + np.var(b))**0.5
                    if geweke_z > geweke_max:
                        self.logger.debug("geweke drift (z=%.1f) detected for parameter '%s'" % (geweke_z, self.paramnames[i]))
                        converged = False
                    
                self.logger.debug("autocorrelation length: tau=%.1f -> %dx lengths" % (max_autocorrlength, num_steps / max_autocorrlength))
                if not converged:
                    break
            
            # merge converged across MPI chains
            if self.use_mpi:
                recv_converged = self.comm.gather(converged, root=0)
                converged = all(self.comm.bcast(recv_converged, root=0))
            
            
            if converged:
                # finally, gelman-rubin diagnostic on chains
                chains = np.asarray([sampler.get_chain(flat=True) for sampler in self.samplers])
                if self.use_mpi:
                    recv_chains = self.comm.gather(chains, root=0)
                    chains = np.concatenate(self.comm.bcast(recv_chains, root=0))

                assert chains.shape == (num_chains, num_steps * num_walkers, self.x_dim), (chains.shape, (num_chains, num_steps * num_walkers, self.x_dim))

                rhat = arviz.rhat(arviz.convert_to_dataset(chains)).x.data
                if self.log:
                    self.logger.info("rhat chain diagnostic: %s (<%.3f is good)", rhat, rhat_max)
                converged = np.all(rhat < rhat_max)

                if self.use_mpi:
                    converged = self.comm.bcast(converged, root=0)
            
            if converged:
                if self.log:
                    self.logger.info("converged!!!")
                break
            
            if self.ncall > max_ncalls:
                if self.log:
                    self.logger.warning("maximum number of likelihood calls reached")
                break

            if self.log:
                self.logger.info("not converged yet at iteration %d after %d evals" % (it + 1, self.ncall))
            #self.logger.error("error at iteration %d" % (it+1))
            last_num_steps = num_steps
            num_steps = int(last_num_steps * 10)
            next_ncalls = ncall_here * 10

            if next_ncalls > max_ncalls:
                if self.log:
                    self.logger.warning("would need more likelihood calls (%d) than maximum (%d) for next step" % (next_ncalls, max_ncalls))
                break
            
            self.logger.debug("expected memory usage: %.2f GiB" % (num_chains * num_steps * num_walkers * self.x_dim * 4 / 1024**3))
            if num_chains * num_steps * num_walkers * self.x_dim * 4 >= 5 * 1024**3:
                if self.log:
                    self.logger.warning("would need too much memory for next step")
                break
            
            if self.log:
                self.logger.info("Running %d MCMC steps ..." % (num_steps))
            ncall_here = 0
            for sampler in self.samplers:
                #chain = sampler.get_chain(flat=True)
                last_samples = sampler.get_chain()[-1, :, :]
                # get a scale small compared to the width of the current posterior
                std = np.clip(last_samples.std(axis=0) / (num_walkers * self.x_dim), a_min=1e-30, a_max=1e-1)
                assert std.shape == (self.x_dim,), std.shape
                
                # sample a point from last chain point
                i = np.ones(num_walkers, dtype=int) * np.random.randint(0, len(last_samples))
                self.logger.info("Starting points chosen: %s, L=%.1f", set(i), L.max())
                # select points
                u = np.clip(last_samples[i, :], 1e-10, 1 - 1e-10)
                # add a bit of noise
                noise = np.random.normal(0, std, size=(num_walkers, self.x_dim))
                u = u + noise
                # avoid border
                u = np.clip(u, 1e-10, 1 - 1e-10)
                assert u.shape == (num_walkers, self.x_dim), (u.shape, (num_walkers, self.x_dim))
                
                if self.log:
                    self.logger.info("Starting at %s +- %s", u.mean(axis=0), u.std(axis=0))
                sampler.reset()
                #self.logger.info("not converged yet at iteration %d" % (it+1))
                
                sampler.run_mcmc(u, last_num_steps, progress=self.log)
                ncall_here += num_walkers
                ncall_here += getattr(sampler, 'ncall', last_num_steps * num_walkers)
                assert ncall_here > 0, ncall_here
                last_samples = sampler.get_chain()[-1, :, :]
                assert last_samples.shape == (num_walkers, self.x_dim), (last_samples.shape, (num_walkers, self.x_dim))
                sampler.reset()
                sampler.run_mcmc(last_samples, num_steps, progress=self.log and progress)
                ncall_here += num_walkers
                ncall_here += getattr(sampler, 'ncall', num_steps * num_walkers)
                assert ncall_here > 0, (ncall_here, getattr(sampler, 'ncall', num_steps * num_walkers))
            
            if self.use_mpi:
                recv_ncall = self.comm.gather(ncall_here, root=0)
                ncall_here = sum(self.comm.bcast(recv_ncall, root=0))
            
            if self.log:
                self.logger.info("Used %d calls in last MCMC run", ncall_here)
            self.ncall += ncall_here

        if self.transform is None:
            eqsamples = np.concatenate([sampler.get_chain(flat=True) for sampler in self.samplers])
        else:
            eqsamples = np.concatenate([self.transform(sampler.get_chain(flat=True)) for sampler in self.samplers])

        if self.use_mpi:
            recv_eqsamples = self.comm.gather(eqsamples, root=0)
            eqsamples = np.concatenate(self.comm.bcast(recv_eqsamples, root=0))
        
        self.results = dict(
            paramnames=self.paramnames,
            posterior=dict(
                mean=eqsamples.mean(axis=0).tolist(),
                stdev=eqsamples.std(axis=0).tolist(),
                median=np.percentile(eqsamples, 50, axis=0).tolist(),
                errlo=np.percentile(eqsamples, 15.8655, axis=0).tolist(),
                errup=np.percentile(eqsamples, 84.1345, axis=0).tolist(),
            ),
            samples=eqsamples,
            ncall = int(self.ncall),
            converged = int(converged),
        )
        return self.results
        
    def print_results(self):
        "" "Give summary of marginal likelihood and parameters."" "
        if self.log:
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

    def plot(self, **kwargs):
        if self.log:
            corner.corner(
                self.results['samples'],
                labels=self.results['paramnames'],
                show_titles=True)
