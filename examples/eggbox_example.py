import argparse
import numpy as np
from numpy import pi, cos, log
import matplotlib.pyplot as plt
import logging, sys

def main(args):

    np.random.seed(2)
    ndim = 2
    paramnames = ['x%d' % (i+1) for i in range(ndim)]
    ndim = len(paramnames)
    
    def loglike(z):
        chi = (cos(z / 2.)).prod(axis=1)
        return (2. + chi)**5

    def transform(x):
        return x * 10 * pi
    
    from autoemcee import ReactiveAffineInvariantSampler, create_logger
    
    create_logger('autoemcee', level=logging.DEBUG)
    sampler = ReactiveAffineInvariantSampler(
        paramnames, loglike, transform=transform, 
        vectorized=True,
        sampler=args.sampler)
    
    sampler.run(
        #num_chains=4,
        #max_improvement_loops=2,
        max_ncalls=1000000,
        min_autocorr_times=1,
    )
    if sampler.mpi_rank != 0:
        return
    sampler.print_results()
    
    from getdist import MCSamples, plots

    samples_g = MCSamples(samples=sampler.results['samples'],
                           names=sampler.results['paramnames'],
                           label='Gaussian')
                           #settings=dict(smooth_scale_2D=1))

    mcsamples = [samples_g]
    g = plots.get_subplot_plotter(width_inch=8)
    g.settings.num_plot_contours = 3
    g.triangle_plot(mcsamples, filled=False, contour_colors=plt.cm.Set1.colors)
    plt.savefig('testeggbox_posterior.pdf', bbox_inches='tight')
    plt.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampler', required=True, choices=['goodman-weare', 'slice'])
    args = parser.parse_args()
    main(args)
