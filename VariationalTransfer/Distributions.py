import numpy as np

class ParametricDistribution:

    def __init__(self, params):
        self._params = params

    def grad_step(self, step):
        self._params = self._params - step

    def set_params(self, params):
        self._params = params

    def get_params(self):
        return np.array(self._params)

    def sample(self):
        raise NotImplementedError

"""
    Normal Posterior with diagonal Covariance
"""
class AnisotropicNormalPosterior(ParametricDistribution):
    def __init__(self, mean=np.array([0]), covariance_diagonal=np.array([1]), min_var=0.01):
        super(AnisotropicNormalPosterior, self).__init__(np.hstack((mean, covariance_diagonal)))
        self._dim = mean.size
        self._min_var = min_var

    def sample(self, nsamples=1):
        midpoint = int(self._params.size/2)
        covar = np.diag(self._params[midpoint: ])
        mean = self._params[0:midpoint]
        return np.random.multivariate_normal(mean, covar, nsamples)

    def set_params(self, params):
        super(AnisotropicNormalPosterior, self).set_params(params)
        self._dim = int(params.size/2)

    def get_mean(self):
        return self._params[0: self._dim]

    def get_variance(self):
        return self._params[self._dim: ]

    def grad_step(self, step):
        self._params = self._params - step
        idx = np.asarray(np.where(self._params < self._min_var))
        self._params[idx[idx >= self._dim]] = self._min_var
        
class NormalPosterior(ParametricDistribution):

    def __init__(self, dim, mean=np.array([0]), covar = np.array([1]), min_var=0.01):
        """
        :param dim: dimension the normal distr
        :param mean: mean vector (numpy.ndarray) [dim]
        :param covar: covariance matrix (numpy.ndarray) [dim]
        :param min_var: value of the minimum variance allowed (limiter value for the params update)
        """
        super(NormalPosterior, self).__init__(np.hstack((mean, np.ravel(covar))))
        self._dim = dim
        self._min_var = min_var
        self._prec = None

    def grad_step(self, step):
        self._params = self._params - step
        self._clip()

    def _clip(self):
        pass #TODO

    def sample(self, nsamples=1):
        covar = np.reshape(self._params[self._dim:], (self._dim, self._dim))
        mean = self._params[0:self._dim]
        return np.random.multivariate_normal(mean, covar, nsamples)

    def set_params(self, params):
        super(NormalPosterior, self).set_params(params)

    def get_mean(self):
        return self._params[0: self._dim]

    def get_covar(self):
        return np.reshape(self._params[self._dim:], (self._dim, self._dim))

    def get_variance(self):
        return self._params[self._dim: ]