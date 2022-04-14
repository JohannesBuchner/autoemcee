import numpy as np
from autoemcee import ReactiveAffineInvariantSampler


def test_run():
    np.random.seed(1)

    paramnames = ['Hinz', 'Kunz']
    def loglike(z):
        assert len(z) == len(paramnames)
        a = -0.5 * (((z - 0.5) / 0.01)**2).sum() + -0.5 * ((z[0] - z[1])/0.01)**2
        loglike.ncalls += 1
        return a
    loglike.ncalls = 0

    def transform(x):
        assert len(x) == len(paramnames)
        return 10. * x - 5.

    sampler = ReactiveAffineInvariantSampler(paramnames, loglike, transform=transform, sampler='goodman-weare')
    r = sampler.run(max_improvement_loops=1)
    
    assert abs(sampler.ncall - loglike.ncalls) == 2, (sampler.ncall, loglike.ncalls)

def test_run_vectorized():
    np.random.seed(1)

    paramnames = ['Hinz', 'Kunz']
    def loglike(z):
        assert len(z[0]) == len(paramnames)
        a = -0.5 * (((z - 0.5) / 0.01)**2).sum(axis=1) + -0.5 * ((z[:,0] - z[:,1])/0.01)**2
        loglike.ncalls += len(z)
        return a

    def transform(x):
        assert len(x[0]) == len(paramnames)
        return 10. * x - 5.

    loglike.ncalls = 0
    sampler = ReactiveAffineInvariantSampler(paramnames, loglike, transform=transform, 
        sampler='goodman-weare', vectorized=True)
    r = sampler.run(max_improvement_loops=2)
    assert 0.95 < sampler.ncall / loglike.ncalls < 1.05, (sampler.ncall, loglike.ncalls)

    loglike.ncalls = 0
    sampler = ReactiveAffineInvariantSampler(paramnames, loglike, transform=transform, 
        sampler='slice', vectorized=True)
    r = sampler.run(max_improvement_loops=2)
    sampler.plot()
    assert 0.95 < sampler.ncall / loglike.ncalls < 1.05, (sampler.ncall, loglike.ncalls)

def test_rosen():
    np.random.seed(1)

    paramnames = ['Hinz', 'Kunz', 'Fuchs', 'Gans', 'Hofer']
    def loglike(theta):
        assert len(theta) == len(paramnames)
        a = theta[:-1]
        b = theta[1:]
        loglike.ncalls += 1
        return -2 * (100 * (b - a**2)**2 + (1 - a)**2).sum()
    loglike.ncalls = 0

    def transform(u):
        assert len(u) == len(paramnames)
        return u * 20 - 10
    
    sampler = ReactiveAffineInvariantSampler(paramnames, loglike, transform=transform, sampler='slice')
    sampler.run(max_improvement_loops=1)
    sampler.print_results()
    assert 0.95 < sampler.ncall / loglike.ncalls < 1.05, (sampler.ncall, loglike.ncalls)
    

if __name__ == '__main__':
    test_run()
    test_run_vectorized()
    test_rosen()
