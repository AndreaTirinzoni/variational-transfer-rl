import numpy as np

class ParametricDistribution:

    def __init__(self, params):
        self._params = params

    def grad_step(self, step):
        self._params = self._params - step

    def set_params(self, params):
        self._params = params

    def get_params(self):
        return self._params

    def sample(self):
        raise NotImplementedError


class AnisotropicNormalPosterior(ParametricDistribution):
    def __init__(self, mean=np.array([0]), covariance_diagonal=np.array([1])):
        super(AnisotropicNormalPosterior, self).__init__(np.hstack((mean, covariance_diagonal)))
        self._dim = mean.size

    def sample(self, nsamples=1):
        midpoint = int(self._params.size/2)
        covar = np.diag(self._params[midpoint: ])
        mean = self._params[0:midpoint]
        return np.random.multivariate_normal(mean, covar, nsamples)
