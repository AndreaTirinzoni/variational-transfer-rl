import numpy as np
from misc import utils
from operators.operator import Operator


class MellowBellmanOperator(Operator):

    def __init__(self, kappa, tau, xi, gamma, state_dim, action_dim):
        self._kappa = kappa
        self._tau = tau
        self._xi = xi
        self._gamma = gamma
        self._state_dim = state_dim
        self._action_dim = action_dim

    def gradient_mm(self, Q, samples, weights=None):
        """General function for computing mellow-max gradients"""
        _, _, _, _, s_prime, absorbing, _ = utils.split_data(samples, self._state_dim, self._action_dim)
        return self._gradient_mm_single(Q, s_prime, absorbing) if weights is None else \
            self._gradient_mm_multi(Q, s_prime, absorbing, weights)

    def _gradient_mm_single(self, Q, s_prime, absorbing):
        """Computes the mellow-max gradient"""
        q_values, q_gradient = Q.value_gradient_actions(s_prime, absorbing)
        sft_Q = utils.softmax(q_values, self._kappa, axis=1)
        mm_grad = np.sum(sft_Q[:, :, np.newaxis] * q_gradient, axis=1)

        return mm_grad

    def _gradient_mm_multi(self, Q, s_prime, absorbing, weights):
        """Computes the mellow-max gradient for different weights"""
        q_values, q_gradient = Q.value_gradient_actions_weights(s_prime, weights, absorbing)
        sft_Q = utils.softmax(q_values, self._kappa, axis=1)
        if q_gradient.ndim == 4:
            mm_grad = np.sum(sft_Q[:, :, :, np.newaxis] * np.transpose(q_gradient, (0,1,3,2)), axis=1)
        else:
            mm_grad = np.sum(sft_Q[:, :, :, np.newaxis] * q_gradient[:, :, np.newaxis, :], axis=1)

        return mm_grad

    def bellman_residual(self, Q, samples, weights=None):
        """General function for computing Bellman residuals"""
        _, _, _, r, s_prime, absorbing, sa = utils.split_data(samples, self._state_dim, self._action_dim)
        return self._bellman_residual_single(Q, r, s_prime, absorbing, sa) if weights is None else \
            self._bellman_residual_multi(Q, r, s_prime, absorbing, sa, weights)

    def _bellman_residual_single(self, Q, r, s_prime, absorbing, sa):
        """Computes the Bellman residuals of given samples"""
        Qs_prime = Q.value_actions(s_prime, absorbing)
        mmQs = utils.mellow_max(Qs_prime, self._kappa)

        return r + self._gamma * mmQs * (1 - absorbing) - Q.value(sa)

    def _bellman_residual_multi(self, Q, r, s_prime, absorbing, sa, weights):
        """Computes the Bellman residuals of a set of samples given a set of weights"""
        Q_values_prime = Q.value_actions_weights(s_prime, weights, absorbing)
        mm = utils.mellow_max(Q_values_prime, self._kappa, axis=1)

        return r[:, np.newaxis] + self._gamma * mm * (1 - absorbing[:, np.newaxis]) - Q.value_weights(sa, weights)

    def gradient_be(self, Q, samples, weights=None):
        """General function for gradients of the Bellman error"""
        _, _, _, _, _, _, sa = utils.split_data(samples, self._state_dim, self._action_dim)
        return self._gradient_be_single(Q, samples, sa) if weights is None else \
            self._gradient_be_multi(Q, samples, sa, weights)

    def _gradient_be_single(self, Q, samples, sa):
        """Computes the Bellman error gradient"""
        br = self.bellman_residual(Q, samples)
        mm_grad = self.gradient_mm(Q, samples)
        q_grad = Q.gradient(sa)
        res_grad = self._xi * self._gamma * mm_grad - q_grad
        be_grad = 2 * np.sum(br[:, np.newaxis] * res_grad * utils.softmax(br ** 2, self._tau)[:, np.newaxis], axis=0)

        return be_grad

    def _gradient_be_multi(self, Q, samples, sa, weights):
        """Computes the gradient of the Bellman error for different weights"""
        br = self.bellman_residual(Q, samples, weights)
        mm_grad = self.gradient_mm(Q, samples, weights)
        q_grad = Q.gradient(sa)
        res_grad = self._xi * self._gamma * mm_grad - q_grad[:, np.newaxis, :]
        be_grad = 2 * np.sum(br[:, :, np.newaxis] * res_grad * utils.softmax(br ** 2, self._tau, axis=0)[:, :, np.newaxis], axis=0)

        return be_grad

    def bellman_error(self, Q, samples, weights=None):
        """General function for computing the Bellman error"""
        return self._bellman_error_single(Q, samples) if weights is None else \
            self._bellman_error_multi(Q, samples, weights)

    def _bellman_error_single(self, Q, samples):
        """Computes the Bellman error"""
        br = self.bellman_residual(Q, samples) ** 2
        return np.sum(utils.softmax(br, self._tau) * br)

    def _bellman_error_multi(self, Q, samples, weights):
        """Computes the Bellman error for each weight"""
        br = self.bellman_residual(Q, samples, weights) ** 2
        errors = np.average(br, axis=0, weights=utils.softmax(br, self._tau, axis=0))
        return errors

    def expected_bellman_error(self, Q, samples, weights):
        """Approximates the expected Bellman error with a finite sample of weights"""
        return np.average(self.bellman_error(Q, samples, weights))
