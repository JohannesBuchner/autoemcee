import argparse
import numpy as np
from numpy import pi, cos, log

def main(args):

    np.random.seed(2)
    ndim = args.x_dim
    paramnames = ['x%d' % (i+1) for i in range(ndim)]
    ndim = len(paramnames)
    
    def loglike(z):
        chi = (cos(z / 2.)).prod(axis=1)
        return (2. + chi)**5

    def transform(x):
        return x * 10 * pi
        
    from autoemcee import ReactiveAffineInvariantSampler
    
    sampler = ReactiveAffineInvariantSampler(paramnames, loglike, transform=transform, vectorized=True)
    
    sampler.run(
        num_global_samples=1000,
        num_chains=4,
        #num_walkers=10,
        max_improvement_loops=40,
    )
    """
    sampler.print_results()
    
    from getdist import MCSamples, plots

    samples_g = MCSamples(samples=sampler.results['samples'],
                           names=sampler.results['paramnames'],
                           label='Gaussian',
                           settings=dict(smooth_scale_2D=3), sampler='nested')

    mcsamples = [samples_g]
    g = plots.get_subplot_plotter(width_inch=8)
    g.settings.num_plot_contours = 3
    g.triangle_plot(mcsamples, filled=False, contour_colors=plt.cm.Set1.colors)
    plt.savefig('testsine_posterior.pdf', bbox_inches='tight')
    plt.close()
    """
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--x_dim', type=int, default=2,
                        help="Dimensionality")

    args = parser.parse_args()
    main(args)
