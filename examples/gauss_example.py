import argparse
import numpy as np
from numpy import pi, sin, log
import matplotlib.pyplot as plt
import logging, sys

def main(args):

    np.random.seed(2)
    ndim = args.x_dim
    sigma = np.logspace(-1, -10, ndim)
    means = np.ones(ndim) * 0.5
    paramnames = ['x%d' % (i+1) for i in range(ndim)]
    ndim = len(paramnames)
    
    def loglike(params):
        logl = (-0.5 * log(2 * pi * sigma.reshape((1, -1))**2) - 0.5 * ((params - means.reshape((1, -1))) / sigma.reshape((1, -1)))**2).sum(axis=1)
        return logl
    
    from autoemcee import ReactiveAffineInvariantSampler, create_logger
    
    create_logger('autoemcee', level=logging.DEBUG)
    sampler = ReactiveAffineInvariantSampler(paramnames, loglike, vectorized=True)
    
    sampler.run(
        #num_global_samples=10000,
        #num_chains=4,
        #num_walkers=10,
        #max_improvement_loops=40,
    )
    if sampler.mpi_rank != 0:
        return
    sampler.print_results()
    
    from getdist import MCSamples, plots

    samples_g = MCSamples(samples=sampler.results['samples'],
                           names=sampler.results['paramnames'],
                           label='Gaussian',
                           settings=dict(smooth_scale_2D=1))

    mcsamples = [samples_g]
    g = plots.get_subplot_plotter(width_inch=8)
    g.settings.num_plot_contours = 3
    g.triangle_plot(mcsamples, filled=False, contour_colors=plt.cm.Set1.colors)
    plt.savefig('testgauss_posterior.pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--x_dim', type=int, default=2,
                        help="Dimensionality")
    parser.add_argument('--sampler', required=True, choices=['goodman-weare', 'slice'])

    args = parser.parse_args()
    main(args)
