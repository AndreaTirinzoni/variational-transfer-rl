import numpy as np
import algorithms

class MellowBellmanOperator(algorithms.BellmanOperator):

    def __init__(self, Q, gamma=0.99, kappa=1e6):

        super(MellowBellmanOperator, self).__init__(Q, gamma)
        self._kappa = kappa

    def __call__(self, mdp_samples):
        s_prime = self._Q.state_dim + self._Q.action_dim + 1
        r = self._Q.state_dim + self._Q.action_dim
        qs = self._mellow_max(mdp_samples[:, s_prime: s_prime + self._state_dim])
        return mdp_samples[:, r] + self._gamma * qs * (1 - mdp_samples[:, -1])

    def compute_gradient_diag_hessian(self, mdp_samples):
        r = self._Q.state_dim + self._Q.action_dim
        s_prime = self._Q.state_dim + self._Q.action_dim + 1
        br = self.bellman_residual(mdp_samples)
        mm_gradient, mm_diag_hess = self._gradient_and_diag_hess_mellow_max(mdp_samples[:, s_prime:-1])
        q_gradient = self._Q.compute_gradient(mdp_samples[:, r])
        b_grad = self._gamma * mm_gradient - q_gradient
        bellman_grad = 2*np.average(br * b_grad)
        bellman_hess = 2*np.average(self._gamma * br * mm_diag_hess + b_grad**2)
        return bellman_grad, bellman_hess

    def _mellow_max(self, states):
        q_values = self._Q.compute_all_actions(states)
        qs = np.sum(np.exp(self._kappa * q_values), axis=1)
        return np.log(qs/q_values.shape[1])/self._kappa

    def _gradient_mellow_max(self, states):
        q_values = self._Q.compute_all_actions(states)
        q_gradient = self._Q.compute_gradient_all_actions(states)
        qs = np.exp(self._kappa * q_values)
        return np.sum(qs * np.exp(q_gradient), axis=1)/qs

    def _diagonal_hessian_mellow_max(self, states):
        q_values = self._Q.compute_all_actions(states)
        q_gradient = self._Q.compute_gradient_all_actions(states)
        qs = np.exp(self._kappa * q_values)
        qs = np.sum(qs * q_gradient**2, axis=1)/qs - (np.sum(qs * q_gradient, axis=1)/qs)**2
        return self._kappa * qs

    def _gradient_and_diag_hess_mellow_max(self, states):
        q_values = self._Q.compute_all_actions(states)
        q_gradient = self._Q.compute_gradient_all_actions(states)
        qs = np.exp(self._kappa * q_values)
        diag_hess = self._kappa * (np.sum(qs * q_gradient**2, axis=1)/qs - (np.sum(qs * q_gradient, axis=1)/qs)**2)
        grad = np.sum(qs * np.exp(q_gradient), axis=1)/qs
        return grad, diag_hess

    def get_Q(self):
        return self._Q;


