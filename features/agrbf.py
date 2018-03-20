import numpy as np
"""
Anisotropic Gaussian RBF Features
"""
class AGaussianRBF:

    def __init__(self, mean, covar, K=2, dims=2):
        """
        :param mean: (np.ndarray) mean vector Kxdim
        :param covar: (np.ndarray)  Covariances vector (Kxdim)xdim
        :param K: number of basis functions
        :param dims: dimension of the input

        """
        assert mean.shape == (K, dims)
        assert covar.shape == (K * dims, dims)
        self._mean = mean
        self._K = K
        self._dims = dims
        self._precision = np.zeros(covar.shape).reshape(dims, dims, K)
        for k in range(self._K):
            self._precision[:, :, k] = np.linalg.inv(covar[k*dims:(k+1)*dims, :])


    def _compute(self, point):

        """
        Computes a feature vector for the point given
        :param point: np.ndarray (n x dim)
        :return: feature vector: np.ndarray
        """
        shape_mean = (int(point.size/self._dims), 1)
        shape_prec = (1,1, shape_mean[0])
        dif = np.tile(self._mean, shape_mean) - np.repeat(point, int(self._K), axis=0)
        dif2 = dif.T[:, np.newaxis, :]
        prec = np.tile(self._precision, shape_prec)
        temp = np.sum(dif2 * prec, axis=0).T
        temp = np.exp(-0.5 * np.sum(temp * dif, axis=1).reshape(shape_mean[0], self._K))
        res = temp/np.sum(temp, axis=1)[:, np.newaxis]

        return res

    def __call__(self, x):
        if x.ndim == 2:
            return self._compute_feature_matrix(x)
        elif x.ndim == 1:
            return self._compute(x[np.newaxis])


    def _compute_feature_matrix(self, data):
        """
        Computes the feature matrix for the dataset passed
        :param data: np.ndarray with a sample per row
        :return: feature matrix (np.ndarray) with feature vector for each row.
        """
        assert data.shape[1] == self._dims
        return self._compute(data)

    def number_of_features(self):
        return self._K

if __name__ == '__main__':
    mean = np.array([[1, 2], [3, 4]])
    print(mean)
    covar = np.vstack((np.eye(2), np.eye(2)))

    rbf = AGaussianRBF(mean, covar)

    data = np.array([[1,2], [3,4], [1.5, 3]])
    data = np.tile(data, (3,1))
    matrix = rbf(data)
    print(rbf(np.array([1, 2])))
    print(matrix)

