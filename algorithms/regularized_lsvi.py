import numpy as np

class RegularizedLSVI:
    """
    Implementation of a regularized (by a normal prior over the parameters of Q) value iteration method.
    """
    @staticmethod
    def solve(features, target, mean=None, covariance=None, prior=True):
        """
        :param features: matrix of feature vectors.
        :param mean: np.ndarray mean of the prior gaussian over the parameters
        :param covariance: np.ndarray covariance matrix of the prior gaussian over parameters
        :param target: np.ndarray with the target values for the fitting
        :return:
        """
        prec = 1

        if prior:
            assert features.shape[1] == mean.size
            assert covariance.shape == (mean.size, mean.size)
            assert target.shape[0] == features.shape[0]
            prec = np.linalg.inv(covariance)
        lamb = 1
        X = np.dot(features.T, features)
        Y = (np.dot(features.T, target) + np.dot(prec, mean)) if prior else np.dot(features.T, target)
        w = np.dot(np.linalg.inv(X + prec) if prior else np.linalg.inv(X + lamb*np.eye(X.shape[0])), Y)

        # u, s, v = np.linalg.svd(features)
        # w = np.dot(v.T[:, : s.shape[0]] @ np.diag(1/s) @ u.T, target)
        return w
