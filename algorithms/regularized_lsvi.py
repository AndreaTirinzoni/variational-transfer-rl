import numpy as np

class RegularizedLSVI:
    """
    Implementation of a regularized (by a normal prior over the parameters of Q) value iteration method.
    """

    @staticmethod
    def solve(features, mean, covariance, target):
        assert features.shape[1] == mean.size
        assert covariance.shape == (mean.size, mean.size)
        assert target.shape[0] == features.shape[0]

        prec = np.invert(covariance)
        X = np.dot(features.T, features)
        return np.dot(np.invert(X + prec), np.dot(features.T, target) + np.dot(prec, mean))
