import numpy as np


class Identity:
    """Identity features"""

    def __init__(self, K, state_dim, n_actions):
        self._K = K
        self._state_dim = state_dim
        self._n_actions = n_actions
        self._idx = [np.arange(0,K, dtype=np.int32) * n_actions + a for a in range(self._n_actions)]

    def _get_mask(self, a):
        mask = np.zeros((a.shape[0], int(self._K * 4)))
        for i in range(self._n_actions):
            mask[np.ix_(a == i, self._idx[i])] = 1
        return mask

    def _compute(self, point):
        mask = self._get_mask(point[:, -1])
        return point[:, :-1] * mask

    def __call__(self, sa):
        """Computes the features of the given array sa N x (state_dim * n_actions + 1)"""
        if sa.ndim == 2:
            return self._compute(sa)
        elif sa.ndim == 1:
            return self._compute(sa[np.newaxis])

    def number_of_features(self):
        return int(self._K * self._n_actions)
