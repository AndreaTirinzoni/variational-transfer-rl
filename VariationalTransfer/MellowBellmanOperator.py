import numpy as np
import VariationalTransfer.BellmanOperator as bo

"""
Optimal Bellman Operator with Mellowmax
"""
class MellowBellmanOperator(bo.BellmanOperator):
    def __init__(self, Q=None, gamma=0.99, kappa=5):
        super(MellowBellmanOperator, self).__init__(Q, gamma)
        self._kappa = kappa

    def __call__(self, mdp_samples):
        s_prime = self._Q.get_statedim() + self._Q.get_actiondim() + 1
        r = self._Q.get_statedim() + self._Q.get_actiondim()
        qs = self._mellow_max(mdp_samples[:, s_prime: s_prime + self._state_dim])
        return mdp_samples[:, r] + self._gamma * qs * (1 - mdp_samples[:, -1])

    def bellman_residual(self, mdp_samples):
        s_prime = self._Q.get_statedim() + self._Q.get_actiondim() + 1
        r = self._Q.get_statedim() + self._Q.get_actiondim()
        q_values = self._Q.compute_all_actions(mdp_samples[:, s_prime:s_prime+self._Q.get_statedim()])
        c = np.max(q_values, axis=1)
        mmQ = self._mellow_max(q_values, c)
        return mdp_samples[:, r] + mmQ - self._Q(mdp_samples[:, 0:r])


    def compute_gradient_diag_hessian(self, mdp_samples):
        r = self._Q.get_statedim() + self._Q.get_actiondim()
        s_prime = self._Q.get_statedim() + self._Q.get_actiondim() + 1
        br = self.bellman_residual(mdp_samples)
        mm_gradient, mm_diag_hess = self._gradient_and_diag_hess_mellow_max(mdp_samples[:, s_prime:-1])
        q_gradient = self._Q.compute_gradient(mdp_samples[:, 0:r])
        q_hessian = self._Q.compute_diag_hessian(mdp_samples[:, 0:r])
        b_grad = self._gamma * mm_gradient - q_gradient
        bellman_grad = 2*np.average(br * b_grad, axis=0)
        bellman_hess = 2*np.average(self._gamma * br * mm_diag_hess + b_grad**2, axis=0) #TODO fix this
        return bellman_grad, bellman_hess

    def _mellow_max(self, q_values):
        qs = np.sum(self._normalized_mm_exp(q_values, np.max(q_values)), axis=1)
        return np.log(qs/q_values.shape[1])/self._kappa + np.max(q_values)

    def _normalized_mm_exp(self, q_values, c=0):
        return np.exp(self._kappa * (q_values-c))

    def _gradient_mellow_max(self, q_values, q_gradient):
        qs = self._normalized_mm_exp(q_values, np.max(q_values))
        qs_sum = np.sum(qs, axis=1)[:, np.newaxis]
        grad = np.sum(qs[:, :, np.newaxis] * q_gradient, axis=1)
        return grad.reshape(grad.shape[0], grad.shape[2])/qs_sum

    def _diagonal_hessian_mellow_max(self, q_values, q_gradient, q_diag_hessian):
        qs = self._normalized_mm_exp(q_values, np.max(q_values))
        qs_sum = np.sum(qs, axis=1)[:, np.newaxis]
        qs = np.sum(qs[:, :, np.newaxis] * (self._kappa * q_gradient**2 + q_diag_hessian) , axis=1)/qs_sum \
             - self._kappa * (np.sum(qs[:, :, np.newaxis] * q_gradient, axis=1)/qs_sum)**2
        return qs.reshape(qs.shape[0], qs.shape[2])

    def _gradient_and_diag_hess_mellow_max(self, states):
        q_values = self._Q.compute_all_actions(states)
        q_gradient = self._Q.compute_gradient_all_actions(states)
        qs = self._normalized_mm_exp(q_values, np.max(q_values))
        qs_sum = np.sum(qs, axis=1)[np.newaxis]
        q_diag_hessian = self._Q.compute_diag_hessian_all_actions(states)
        diag_hess = np.sum(qs[:, :, np.newaxis] * (self._kappa * q_gradient**2 + q_diag_hessian), axis=1)/qs_sum \
             - self._kappa * (np.sum(qs[:, :, np.newaxis] * q_gradient, axis=1)/qs_sum)**2
        grad = np.sum(qs[:, :, np.newaxis] * q_gradient, axis=1)/qs_sum
        return grad, diag_hess


"""
Mellow Bellman Operator optimized for Linear Q regressor
"""
class LinearQMellowBellman(MellowBellmanOperator):
    
    def compute_gradient_diag_hessian(self, mdp_samples, weights=None):
        if weights is None:
            return super(LinearQMellowBellman, self).compute_gradient_diag_hessian(mdp_samples)
        else:
            r = self._Q.get_statedim() + self._Q.get_actiondim()
            s_prime = self._Q.get_statedim() + self._Q.get_actiondim() + 1
            br = self.bellman_residual(mdp_samples, weights)
            br = br.reshape(br.shape[0], 1, br.shape[1])
            mm_gradient, mm_diag_hess = self._gradient_and_diag_hess_mellow_max(mdp_samples[:, s_prime:s_prime + self._Q.get_statedim()], weights=weights)
            q_gradient = self._Q.compute_gradient(mdp_samples[:, 0:r])
            b_grad = self._gamma * mm_gradient - q_gradient[:, :, np.newaxis]
            bellman_grad = 2 * np.average(br * b_grad, axis=0)
            bellman_hess = 2 * np.average(self._gamma * br * mm_diag_hess + b_grad ** 2, axis=0)
            return bellman_grad, bellman_hess

    def bellman_residual(self, mdp_samples, weights=None):
        if weights is None:
            return super(LinearQMellowBellman, self).bellman_residuals(mdp_samples)
        else:
            s_prime = self._Q.get_statedim() + self._Q.get_actiondim() + 1
            r = self._Q.get_statedim() + self._Q.get_actiondim()
            feats_sprime = self._Q.compute_gradient_all_actions(mdp_samples[:, s_prime:s_prime + self._Q.get_statedim()])
            feats = self._Q.compute_gradient(mdp_samples[:, 0: self._Q.get_statedim() + self._Q.get_actiondim()])
            Q = feats @ weights.T
            nacts = self._Q.actions.size

            # Compute normalization constants
            q = np.dot(feats_sprime.reshape(feats_sprime.shape[0] * feats_sprime.shape[1], feats_sprime.shape[2]), \
                                            weights.T)
            c = np.max(q.reshape(feats_sprime.shape[0], feats_sprime.shape[1], weights.shape[0]), axis=1)
            mmQ = self._normalized_mm_exp(feats_sprime, c, weights=weights)
            # c = np.dot(c, weights.T)
            mmQ = np.log(np.sum(mmQ, axis=1)/nacts)/self._kappa + c
            mmQ = mmQ.reshape(Q.shape)

            return mdp_samples[:, r, np.newaxis] + self._gamma * mmQ - Q

    def _gradient_and_diag_hess_mellow_max(self, states, weights=None):
        if weights is None:
            return super._gradient_and_diag_hess_mellow_max(states)
        else:
            q_gradient = self._Q.compute_gradient_all_actions(states) # features

            # Compute normalization constants
            q = np.dot(q_gradient.reshape(q_gradient.shape[0] * q_gradient.shape[1], q_gradient.shape[2]), \
                       weights.T)
            c = np.max(q.reshape(q_gradient.shape[0], q_gradient.shape[1], weights.shape[0]), axis=1)

            qs = self._normalized_mm_exp(q_gradient, c=c, weights=weights)
            qs_sum = np.sum(qs, axis=1)
            qs_sum = qs_sum.reshape(qs_sum.shape[0], 1, qs_sum.shape[1])
            q_gradient = q_gradient[:,:,:,np.newaxis]
            qs = qs.reshape(qs.shape[0], qs.shape[1], 1, qs.shape[2])
            diag_hess = self._kappa * (np.sum(qs * q_gradient**2, axis=1)/qs_sum \
                                       - (np.sum(qs * q_gradient, axis=1)/qs_sum)**2)
            grad = np.sum(qs * q_gradient, axis=1)/qs_sum
            return grad, diag_hess

    def _normalized_mm_exp(self, q_values, c=0, weights=None):
        if weights is None:
            return super._normalized_mm_ex(q_values, c)
        else:
            # q_values are only features
            nacts = self._Q.actions.size
            # t = q_values - c[:, np.newaxis]
            qs = q_values.reshape(q_values.shape[0] * q_values.shape[1], q_values.shape[2])
            q = np.dot(qs, weights.T)
            qs = np.exp(self._kappa * (q - np.repeat(c, q_values.shape[1], axis=0)))
            return qs.reshape(q_values.shape[0], q_values.shape[1], weights.shape[0])

