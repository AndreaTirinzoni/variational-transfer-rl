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


def build_features_gw(gw_size, n_basis, n_actions, state_dim, action_dim):
    """Create ARBF for gridworld"""
    # Number of features
    K = n_basis ** 2 * n_actions
    # Build the features
    x = np.linspace(0, gw_size, n_basis)
    y = np.linspace(0, gw_size, n_basis)
    a = np.linspace(0, n_actions - 1, n_actions)
    mean_x, mean_y, mean_a = np.meshgrid(x, y, a)
    mean = np.hstack((mean_x.reshape(K, 1), mean_y.reshape(K, 1), mean_a.reshape(K, 1)))
    assert mean.shape == (K, 3)

    state_var = (gw_size / (n_basis - 1) / 3) ** 2
    action_var = 0.01 ** 2
    covar = np.eye(state_dim + action_dim)
    covar[0:state_dim, 0:state_dim] *= state_var
    covar[-1, -1] *= action_var
    assert covar.shape == (3, 3)
    covar = np.tile(covar, (K, 1))
    assert covar.shape == (3 * K, 3)

    return AGaussianRBF(mean, covar, K=K, dims=state_dim + action_dim)


def build_features_gw_state(gw_size, n_basis, state_dim):
    """Create ARBF for gridworld as functions of the state only"""
    # Number of features
    K = n_basis ** 2
    # Build the features
    x = np.linspace(0, gw_size, n_basis)
    y = np.linspace(0, gw_size, n_basis)
    mean_x, mean_y = np.meshgrid(x, y)
    mean = np.hstack((mean_x.reshape(K, 1), mean_y.reshape(K, 1)))

    state_var = (gw_size / (n_basis - 1) / 3) ** 2
    covar = np.eye(state_dim) * state_var
    covar = np.tile(covar, (K, 1))

    return AGaussianRBF(mean, covar, K=K, dims=state_dim)


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

