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
        q_gradient = self._Q.compute_gradient(mdp_samples[:, 0:r])
        b_grad = self._gamma * mm_gradient - q_gradient
        bellman_grad = 2*np.average(br * b_grad, axis=0)
        bellman_hess = 2*np.average(self._gamma * br * mm_diag_hess + b_grad**2, axis=0)
        return bellman_grad, bellman_hess

    def _mellow_max(self, q_values):
        qs = self._normalized_mm_exp(q_values)
        return np.log(qs/q_values.shape[1])/self._kappa

    def _normalized_mm_exp(self, q_values, c=0):
        # TODO: add normalization to avoid overflow
        return np.exp(self._kappa * q_values)

    def _gradient_mellow_max(self, q_values, q_gradient):
        qs = self._normalized_mm_exp(q_values)
        qs_sum = np.sum(qs, axis=1)[:, np.newaxis]
        grad = np.sum(qs[:, :, np.newaxis] * q_gradient, axis=1)
        return grad.reshape(grad.shape[0], grad.shape[2])/qs_sum

    def _diagonal_hessian_mellow_max(self, q_values, q_gradient, q_diag_hessian):
        qs = self._normalized_mm_exp(q_values)
        qs_sum = np.sum(qs, axis=1)[:, np.newaxis]
        qs = np.sum(qs[:, :, np.newaxis] * (self._kappa * q_gradient**2 + q_diag_hessian) , axis=1)/qs_sum \
             - self._kappa * (np.sum(qs[:, :, np.newaxis] * q_gradient, axis=1)/qs_sum)**2
        return qs.reshape(qs.shape[0], qs.shape[2])

    def _gradient_and_diag_hess_mellow_max(self, states):
        q_values = self._Q.compute_all_actions(states)
        q_gradient = self._Q.compute_gradient_all_actions(states)
        qs = self._normalized_mm_exp(q_values)
        qs_sum = np.sum(qs, axis=1)[np.newaxis]
        q_diag_hessian = self._Q.compute_diag_hessian_all_actions(states)
        diag_hess = np.sum(qs[:, :, np.newaxis] * (self._kappa * q_gradient**2 + q_diag_hessian), axis=1)/qs_sum \
             - self._kappa * (np.sum(qs[:, :, np.newaxis] * q_gradient, axis=1)/qs_sum)**2
        grad = np.sum(qs[:, :, np.newaxis] * q_gradient, axis=1)/qs_sum
        return grad, diag_hess

    def get_Q(self):
        return self._Q

class LinearQMellowBellman(MellowBellmanOperator):
    
    def compute_gradient_diag_hessian(self, mdp_samples, weights=None):
        if weights is None:
            return super(LinearQMellowBellman, self).compute_gradient_diag_hessian(mdp_samples)
        else:
            r = self._Q.state_dim + self._Q.action_dim
            s_prime = self._Q.state_dim + self._Q.action_dim + 1
            br = self.bellman_residual(mdp_samples, weights)
            br = br.reshape(br.shape[0], 1, br.shape[1])
            mm_gradient, mm_diag_hess = self._gradient_and_diag_hess_mellow_max(mdp_samples[:, s_prime:s_prime + self._Q.state_dim], weights=weights)
            q_gradient = self._Q.compute_gradient(mdp_samples[:, 0:r])
            b_grad = self._gamma * mm_gradient - q_gradient[:, :, np.newaxis]
            bellman_grad = 2 * np.average(br * b_grad, axis=0)
            bellman_hess = 2 * np.average(self._gamma * br * mm_diag_hess + b_grad ** 2, axis=0)
            return bellman_grad, bellman_hess

    def bellman_residuals(self, mdp_samples, weights=None):
        if weights is None:
            return super.bellman_residuals(mdp_samples)
        else:
            s_prime = self._Q.state_dim + self._Q.action_dim + 1
            r = self._Q.state_dim + self._Q.action_dim
            feats_sprime = self._Q.compute_gradient_all_actions(mdp_samples[:, s_prime:s_prime + self._Q.state_dim])
            feats = self._Q.compute_gradient(mdp_samples[:, 0: self._Q.state_dim + self._Q.action_dim])
            Q = feats @ weights.T
            nacts = self._Q._actions.size
            mmQ = self._normalized_mm_exp(feats_sprime, weights=weights)
            mmQ = np.log(np.sum(mmQ, axis=1)/nacts)/self._kappa
            mmQ = mmQ.reshape(Q.shape)

            return mdp_samples[:, r, np.newaxis] + self._gamma * mmQ - Q

    def _gradient_and_diag_hess_mellow_max(self, states, weights=None):
        if weights is None:
            return super._gradient_and_diag_hess_mellow_max(states)
        else:
            q_gradient = self._Q.compute_gradient_all_actions(states) # features
            qs = self._normalized_mm_exp(q_gradient, weights=weights)
            qs_sum = np.sum(qs, axis=1)
            qs_sum = qs_sum.reshape(qs_sum.shape[0], 1, qs_sum.shape[1])
            q_gradient = q_gradient[:,:,:,np.newaxis]
            qs = qs.reshape(qs.shape[0], qs.shape[1], 1, qs.shape[2])
            diag_hess = self._kappa * (np.sum(qs * q_gradient**2, axis=1)/qs_sum \
                                       - (np.sum(qs * q_gradient, axis=1)/qs_sum)**2)       #TODO fix
            grad = np.sum(qs * q_gradient, axis=1)/qs_sum
            return grad, diag_hess

    def _normalized_mm_exp(self, q_values, c=0, weights=None):
        if weights is None:
            return super._normalized_mm_ex(q_values,c)
        else:
            # q_values are only features
            nacts = self._Q._actions.size
            qs = q_values.reshape(q_values.shape[0] * q_values.shape[1], q_values.shape[2])
            qs = np.exp(self._kappa * np.dot(qs, weights.T)) # TODO: Add normalization to avoid overflow
            return qs.reshape(q_values.shape[0], q_values.shape[1], weights.shape[0])