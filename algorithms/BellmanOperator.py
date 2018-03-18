import numpy as np

class BellmanOperator:

    """
    Optimal Bellman Operator
    """

    def __init__ (self, Q, gamma=0.99):
        self._Q = Q
        self._gamma = gamma

    def __call__(self, mdp_samples):
        s_prime = self._Q.state_dim + self._Q.action_dim + 1
        r = self._Q.state_dim + self._Q.action_dim
        a = self._Q.state_dim
        qs = np.max(self._Q.compute_all_actions(mdp_samples[:, s_prime: s_prime + self._state_dim]), axis=1)
        return mdp_samples[:, r] + self._gamma * qs * (1 - mdp_samples[:, -1])

    def bellman_residual(self, mdp_samples):
        s = self._Q.state_dim
        a = self._Q.state_dim
        r = self._Q.state_dim + self._Q._action_dim
        residuals = -self._Q(mdp_samples[:, 0:r]) + self.__call__(mdp_samples)
        return residuals

