import numpy as np

class BellmanOperator:

    """
    Optimal Bellman Operator
    """

    def __init__ (self, Q=None, gamma=0.99):
        self._Q = Q
        self._gamma = gamma

    def __call__(self, mdp_samples):
        s_prime = self._Q.get_statedim() + self._Q.get_actiondim() + 1
        r = self._Q.get_statedim() + self._Q.get_actiondim()
        a = self._Q.get_statedim()
        qs = np.max(self._Q.compute_all_actions(mdp_samples[:, s_prime: s_prime + self._Q.get_statedim()]), axis=1)
        return mdp_samples[:, r] + self._gamma * qs * (1 - mdp_samples[:, -1])

    def bellman_residual(self, mdp_samples):
        s = 0
        a = self._Q.get_statedim()
        r = self._Q.get_statedim() + self._Q.get_actiondim()
        residuals = -self._Q(mdp_samples[:, 0:r]) + self.__call__(mdp_samples)
        return residuals

    def set_Q(self, Q):
        self._Q = Q

    def get_Q(self):
        return self._Q
