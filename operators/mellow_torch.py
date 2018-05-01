import utils
from operators.operator import Operator
import numpy as np
import torch



def mellow_max(a, kappa, axis=0):
    """ Torch implementation of Mellowmax """
    mx, _ = torch.max(a, dim=axis, keepdim=True)
    return (kappa * (a - mx)).exp().sum(dim=axis, keepdim=False).log() + mx.squeeze(dim=axis)


class MellowBellmanOperator(Operator):

    def __init__(self, kappa, tau, xi, gamma, state_dim, action_dim):
        self._kappa = kappa
        self._tau = tau
        self._xi = xi
        self._gamma = gamma
        self._state_dim = state_dim
        self._action_dim = action_dim

    def bellman_residual(self, Q, samples, weights=None):
        """General function for computing Bellman residuals"""
        _, _, _, r, s_prime, absorbing, sa = utils.split_data(samples, self._state_dim, self._action_dim)
        return self._bellman_residual_single(Q, r, s_prime, absorbing, sa) if weights is None else \
            self._bellman_residual_multi(Q, r, s_prime, absorbing, sa, weights)

    def _bellman_residual_single(self, Q, r, s_prime, absorbing, sa):
        """Computes the Bellman residuals of given samples"""
        Qs_prime = Q.value_actions(s_prime, absorbing)
        mmQs = utils.mellow_max(Qs_prime, self._kappa, axis=1)
        return r + self._gamma * mmQs * (1 - absorbing) - Q.value(sa)

    def _bellman_residual_multi(self, Q, r, s_prime, absorbing, sa, weights):
        """Computes the Bellman residuals of a set of samples given a set of weights"""
        Q_values_prime = Q.value_actions_weights(s_prime, weights, absorbing)
        mm = utils.mellow_max(Q_values_prime, self._kappa, axis=1)
        return r[:, None] + self._gamma * mm * (1 - absorbing[:, None]) - Q.value_weights(sa, weights)

    def _bellman_residual_surrogate(self, Q, samples, weights=None):
        _, _, _, r, s_prime, absorbing, sa = utils.split_data(samples, self._state_dim, self._action_dim)
        if weights is None:
            Qs_prime = Q.value_actions(s_prime, absorbing, grad_required=True)
            mmQs = mellow_max(Qs_prime, self._kappa, axis=1)
            r = torch.from_numpy(r)
            absorbing = torch.from_numpy(absorbing)
            qval = Q.value(sa, grad_required=True)
        else:
            Qs_prime = Q.value_actions_weights(s_prime, weights, absorbing, grad_required=True)
            mmQs = mellow_max(Qs_prime, self._kappa, axis=1)
            r = torch.from_numpy(r).unsqueeze(1)
            absorbing = torch.from_numpy(absorbing).unsqueeze(1)
            qval = Q.value_weights(sa, grad_required=True)

        mean_weight = (r + self._gamma * mmQs * (1 - absorbing) - qval).detach()
        return 2 * mean_weight * (r + self._xi * self._gamma * mmQs * (1-absorbing) - qval)


    def gradient_be(self, Q, samples, weights=None):
        """General function for gradients of the Bellman error"""
        Q.gradient(prepare=True)  # zeroes out old grads.
        if weights is not None:     # TODO fix: some variable is being modified in-place
            be = (self._bellman_residual_surrogate(Q, samples, weights) ** 2).mean(dim=0)
            m = torch.eye(weights.shape[0], dtype=torch.float64)
            J = np.zeros(weights.shape)
            for k in range(weights.shape[0]):
                be.backward(m[:,k], retain_graph=True)
                J[k, :] = Q.gradient()
            return J
        else:

            be = (self._bellman_residual_surrogate(Q, samples)).mean()
            be.backward()
            return Q.gradient()

    def bellman_error(self, Q, samples, weights=None, grad_required=False):
        """General function for computing the Bellman error"""
        return self._bellman_error_single(Q, samples) if weights is None else \
            self._bellman_error_multi(Q, samples, weights)

    def _bellman_error_single(self, Q, samples, grad_required=False):
        """Computes the Bellman error"""
        br = self.bellman_residual(Q, samples) ** 2
        return br.sum()

    def _bellman_error_multi(self, Q, samples, weights, grad_required=False):
        """Computes the Bellman error for each weight"""
        br = self.bellman_residual(Q, samples, weights) ** 2
        errors = np.average(br, axis=0)
        return errors

    def expected_bellman_error(self, Q, samples, weights, grad_required=False):
        """Approximates the expected Bellman error with a finite sample of weights"""
        return np.average(self.bellman_error(Q, samples, weights))


if __name__ == "__main__":
    from approximators.mlp_torch import MLPQFunction

    l1 = 32
    l2 = 0
    kappa = 100.
    tau = 0.
    xi = 0.
    gamma = 0.99
    state_dim = 3
    action_dim = 1
    n_actions = 3


    # Create BellmanOperator
    operator = MellowBellmanOperator(kappa, tau, xi, gamma, state_dim, action_dim)
    # Create Q Function
    layers = [l1]
    if l2 > 0:
        layers.append(l2)
    Q = MLPQFunction(state_dim, n_actions, layers=layers)

    w = Q._w

    weights = torch.randn(5, w.shape[0], requires_grad=True)

    samples = np.random.randn(10, 1 + state_dim + action_dim + 1 + state_dim + 1)
    samples[:, -1] = 0.


    val = Q.value_weights(samples[:, 1:1+state_dim+action_dim] , weights.detach().numpy())
    Q.update_weights(w)
    g = operator.gradient_be(Q, samples)


    val = Q.value(samples[:, 1:1+state_dim+action_dim])
    g = operator.gradient_be(Q, samples, weights.detach().numpy())
