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

    """
        This gradient/Hessian only differentiates with respect to the Q function.
    """
    def compute_gradient_diag_hessian(self, mdp_samples):
        s = 0
        a = self._Q.get_statedim()
        r = self._Q.get_statedim() + self._Q.get_actiondim()
        br = self.bellman_residual(mdp_samples)
        q_gradient = self._Q.compute_gradient(mdp_samples[:, 0:r])
        q_hessian = self._Q.compute_diag_hessian(mdp_samples[:, 0:r])
        b_grad = -q_gradient
        bellman_grad = 2 * np.average(br * b_grad, axis=0)
        bellman_hess = 2 * np.average(q_gradient**2 + br * q_hessian, axis=0)
        return bellman_grad, bellman_hess

    def compute_gradient_hessian(self, mdp_samples):
        s = 0
        a = self._Q.get_statedim()
        r = self._Q.get_statedim() + self._Q.get_actiondim()
        br = self.bellman_residual(mdp_samples)
        q_gradient = self._Q.compute_gradient(mdp_samples[:, 0:r])
        q_hessian = self._Q.compute_hessian(mdp_samples[:, 0:r])
        b_grad = -q_gradient
        bellman_grad = 2 * np.average(br * b_grad, axis=0)
        F = b_grad[:, np.newaxis, :] * b_grad[np.newaxis, :, :]
        bellman_hess = 2 * np.average(F + br.T[:, np.newaxis] * q_hessian, axis=0)
        return bellman_grad, bellman_hess

    def set_Q(self, Q):
        self._Q = Q

    def get_Q(self):
        return self._Q


class LinearQBellmanOperator(BellmanOperator):

    def bellman_residual(self, mdp_samples, weights=None):
        if weights is None:
            return super(LinearQBellmanOperator, self).bellman_residual(mdp_samples)
        else:
            s = 0
            a = self._Q.get_statedim()
            r = self._Q.get_statedim() + self._Q.get_actiondim()
            sprime = r + 1
            feats_sprime = self._Q.compute_gradient_all_actions(mdp_samples[:, sprime:sprime + self._Q.get_statedim()])
            feats = self._Q.compute_gradient(mdp_samples[:, 0:r])
            Q = feats @ weights.T
            Q *= (1-mdp_samples[:, -1])[:, np.newaxis]
            qs = feats_sprime.reshape(feats_sprime.shape[0] * feats_sprime.shape[1], feats_sprime.shape[2]) @ weights.T
            qs = np.max(qs.reshape(feats_sprime.shape[0], feats_sprime.shape[1], weights.shape[0]), axis=1)

            return mdp_samples[:, r, np.newaxis] + self._gamma * qs * (1 - mdp_samples[:, -1])[:, np.newaxis] - Q

    def compute_gradient_diag_hessian(self, mdp_samples, weights=None):
        if weights is None:
            super(LinearQBellmanOperator, self).compute_gradient_diag_hessian(mdp_samples)
        else:
            s = 0
            a = self._Q.get_statedim()
            r = self._Q.get_statedim() + self._Q.get_actiondim()
            sprime = r + 1
            br = self.bellman_residual(mdp_samples, weights)
            b_grad = -self._Q.compute_gradient(mdp_samples[:, 0:r])
            grad = 2 * np.average(br[:, np.newaxis] * b_grad[:, :, np.newaxis], axis=0)
            diag_hess = 2 * np.average(b_grad**2, axis=0)[:, np.newaxis]

            return grad, diag_hess

    def compute_gradient_hessian(self, mdp_samples, weights=None):
        if weights is None:
            super(LinearQBellmanOperator, self).compute_gradient_hessian(mdp_samples)
        else:
            s = 0
            a = self._Q.get_statedim()
            r = self._Q.get_statedim() + self._Q.get_actiondim()
            sprime = r + 1
            br = self.bellman_residual(mdp_samples, weights)
            b_grad = -self._Q.compute_gradient(mdp_samples[:, 0:r]).T
            grad = 2 * np.average(br[:, np.newaxis] * b_grad.T[:, :, np.newaxis], axis=0)
            hess = 2 * np.average(b_grad[:, np.newaxis, :] * b_grad[np.newaxis, :, :], axis=2)

            return grad, hess[:, :, np.newaxis]

