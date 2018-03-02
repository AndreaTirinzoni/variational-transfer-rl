import numpy as np

class RegularizedLSVI:
    """
    Implementation of a regularized (by a normal prior over the parameters of Q) value iteration method.
    """
    @staticmethod
    def solve(features, mean, covariance, target, prior=True):
        """
        :param features: matrix of feature vectors.
        :param mean: np.ndarray mean of the prior gaussian over the parameters
        :param covariance: np.ndarray covariance matrix of the prior gaussian over parameters
        :param target: np.ndarray with the target values for the fitting
        :return:
        """
        assert features.shape[1] == mean.size
        assert covariance.shape == (mean.size, mean.size)
        assert target.shape[0] == features.shape[0]
        prec = np.linalg.inv(covariance)
        X = np.dot(features.T, features)
        Y = (np.dot(features.T, target) + np.dot(prec, mean)) if prior else np.dot(features.T, target)
        return np.dot(np.linalg.inv(X + prec), Y)