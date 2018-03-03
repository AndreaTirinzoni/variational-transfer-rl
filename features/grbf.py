import numpy as np

class GaussianRBF:

    def __init__(self, mean, variance, K=2, dims=2):
        """
        :param mean: (np.ndarray) mean vector Kxdim
        :param variance: (np.ndarray)  variance vector Kx1
        :param K: number of basis functions
        :param dims: dimension of the input

        """
        assert mean.shape == (K, dims)
        assert variance.shape == (K,)


        self._K = K
        self._mean = mean
        self._dims = dims
        self._var = variance

    def _compute(self, point):

        """
        Computes a feature vector for the point given
        :param point: np.ndarray (dim)
        :return: feature vector: np.ndarray
        """
        val = []

        for k in range(self._K):
            dif = self._mean[k, :] - point
            dif = np.dot(dif, dif)
            val.append(np.exp(-dif/self._var[k]))
        f = np.asarray(val, order='F')
        return f

    def __call__(self, x):
        if x.ndim == 2:
            return self._compute_feature_matrix(x)
        elif x.ndim == 1:
            return self._compute(x)


    def _compute_feature_matrix(self, data):
        """
        Computes the feature matrix for the dataset passed
        :param data: np.ndarray with a sample per row
        :return: feature matrix (np.ndarray) with feature vector for each row.
        """
        assert data.shape[1] == self._dims
        features = []
        for x in range(data.shape[0]):
            features.append(self._compute(data[x, :]))

        return np.asarray(features)


if __name__ == '__main__':
    mean = np.array([[1, 2], [3, 4]])
    var = np.array([3, 8])

    rbf = GaussianRBF(mean, var)

    data = np.random.random((100, 2))

    matrix = rbf(data)
    print(matrix)
    print(rbf(np.array([1, 2])))
