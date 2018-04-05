import numpy as np
import VariationalTransfer.BellmanOperator as bo

"""
Optimal Bellman Operator with Mellowmax
"""
class MellowBellmanOperator(bo.BellmanOperator):

    def __init__(self, Q=None, gamma=0.99, kappa=5, tau=0., xi = 1.0):
        super(MellowBellmanOperator, self).__init__(Q, gamma)
        self._kappa = kappa
        self._tau = tau
        self._xi = xi

    def __call__(self, mdp_samples):
        s_prime = self._Q.get_statedim() + self._Q.get_actiondim() + 1
        r = self._Q.get_statedim() + self._Q.get_actiondim()
        qs = self._mellow_max(mdp_samples[:, s_prime: s_prime + self._state_dim])
        return mdp_samples[:, r] + self._gamma * qs * (1 - mdp_samples[:, -1])

    def bellman_residual(self, mdp_samples):
        s_prime = self._Q.get_statedim() + self._Q.get_actiondim() + 1
        r = self._Q.get_statedim() + self._Q.get_actiondim()
        q_values = self._Q.compute_all_actions(mdp_samples[:, s_prime:s_prime+self._Q.get_statedim()])
        mmQ = self._mellow_max(q_values)
        return mdp_samples[:, r] + self._gamma * mmQ * (1 - mdp_samples[:, -1]) - self._Q(mdp_samples[:, 0:r])

    def compute_gradient_diag_hessian(self, mdp_samples):
        r = self._Q.get_statedim() + self._Q.get_actiondim()
        s_prime = self._Q.get_statedim() + self._Q.get_actiondim() + 1
        br = self.bellman_residual(mdp_samples)
        mm_gradient, mm_diag_hess = self._gradient_and_diag_hess_mellow_max(mdp_samples[:, s_prime:-1])
        q_gradient = self._Q.compute_gradient(mdp_samples[:, 0:r])
        q_hessian = self._Q.compute_diag_hessian(mdp_samples[:, 0:r])
        b_grad = self._xi * self._gamma * mm_gradient - q_gradient

        if self._tau == 0.:
            bellman_grad = 2*np.average(br * b_grad, axis=0)
            bellman_hess = 2*np.average(br * (self._xi * self._gamma * mm_diag_hess - q_hessian) + b_grad**2, axis=0)
        else:
            soft_w = self._softmax(br**2)
            bellman_grad = 2 * np.sum(soft_w * br * b_grad, axis=0)
            bellman_hess = 2 * np.sum(soft_w * br * (self._xi * self._gamma * mm_diag_hess - q_hessian) + b_grad ** 2, axis=0)


        return bellman_grad, bellman_hess

    def compute_gradient_hessian(self, mdp_samples):
        r = self._Q.get_statedim() + self._Q.get_actiondim()
        s_prime = self._Q.get_statedim() + self._Q.get_actiondim() + 1
        br = self.bellman_residual(mdp_samples) # s
        mm_gradient, mm_hess = self._gradient_and_hessian_mellow_max(mdp_samples[:, s_prime:-1])       # s x w, s x w x w
        q_gradient = self._Q.compute_gradient(mdp_samples[:, 0:r]) # s x w
        q_hessian = self._Q.compute_hessian(mdp_samples[:, 0:r]) # s x w x w
        b_grad = self._xi * self._gamma * mm_gradient - q_gradient     # s x w

        if self._tau == 0.:
            bellman_grad = 2 * np.average(br[:, np.newaxis] * b_grad, axis=0)
            bellman_hess = 2 * np.average(br[:, np.newaxis, np.newaxis] * (self._xi * self._gamma * mm_hess - q_hessian)
                                          + b_grad[:, :, np.newaxis] * b_grad[:, np.newaxis, :], axis=0) # w x w
        else:
            soft_w = self._softmax(br**2)
            bellman_grad = 2 * np.sum((soft_w * br)[:, np.newaxis] * b_grad, axis=0)
            bellman_hess = 2 * np.sum(soft_w[:, np.newaxis, np.newaxis] * (br[:, np.newaxis, np.newaxis] * (self._xi * self._gamma * mm_hess - q_hessian)
                                          + b_grad[:, :, np.newaxis] * b_grad[:, np.newaxis, :]), axis=0)

        return bellman_grad, bellman_hess

    def _mellow_max(self, q_values):
        c = np.max(q_values, axis=1)
        qs = np.sum(self._normalized_mm_exp(q_values, c), axis=1)
        return np.log(qs/q_values.shape[1])/self._kappa + c

    def _normalized_mm_exp(self, q_values, c=0):
        return np.exp(self._kappa * (q_values-c[:, np.newaxis]))

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
        qs_sum = np.sum(qs, axis=1)[:, np.newaxis]
        q_diag_hessian = self._Q.compute_diag_hessian_all_actions(states)
        diag_hess = np.sum(qs[:, :, np.newaxis] * (self._kappa * q_gradient**2 + q_diag_hessian), axis=1)/qs_sum \
             - self._kappa * (np.sum(qs[:, :, np.newaxis] * q_gradient, axis=1)/qs_sum)**2
        grad = np.sum(qs[:, :, np.newaxis] * q_gradient, axis=1)/qs_sum
        return grad, diag_hess

    def _gradient_and_hessian_mellow_max(self, states):
        q_values = self._Q.compute_all_actions(states)          # s x a
        q_gradient = self._Q.compute_gradient_all_actions(states) * (1-states[:,-1])[:, np.newaxis, np.newaxis]   # s x a x w
        qs = self._normalized_mm_exp(q_values, np.max(q_values, axis=1))    # s x a
        qs_sum = np.sum(qs, axis=1)[:, np.newaxis] # s x 1
        q_hessian = self._Q.compute_hessian_all_actions(states)       # s x a x w x w
        grad = np.sum(qs[:, :, np.newaxis] * q_gradient, axis=1) / qs_sum  # s x w

        outer_mm_grad = grad[:, :, np.newaxis] * grad[:, np.newaxis, :]  # s x w x w
        outer_q_grad = q_gradient[:, :, :, np.newaxis] * q_gradient[:, :, np.newaxis, :] # s x a x w x w

        hess = (self._kappa * np.sum(qs[:, :, np.newaxis, np.newaxis] * outer_q_grad, axis=1) \
        + np.sum(qs[:, :, np.newaxis, np.newaxis] * q_hessian, axis=1))/qs_sum[:, :, np.newaxis] \
        - outer_mm_grad  # s x w x w

        return grad, hess


    def _softmax(self, X):
        mx = np.max(X)
        num = np.exp(self._tau * (X - mx))
        return num / np.sum(num)


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
            br = self.bellman_residual(mdp_samples, weights) #
            mm_gradient, mm_diag_hess = self._gradient_and_diag_hess_mellow_max(mdp_samples[:, s_prime:s_prime + self._Q.get_statedim()], weights=weights)
            q_gradient = self._Q.compute_gradient(mdp_samples[:, 0:r])
            b_grad = (self._xi * self._gamma * mm_gradient) - q_gradient[:, :, np.newaxis]

            if self._tau == 0.0:
                bellman_grad = 2 * np.average(br[:, np.newaxis] * b_grad, axis=0)
                bellman_hess = 2 * np.average((self._xi * self._gamma * br[:, np.newaxis] * mm_diag_hess) + b_grad ** 2, axis=0)
            else:
                soft_w = self._softmax(br ** 2)
                bellman_grad = 2 * np.sum(soft_w[:, np.newaxis] * br[:, np.newaxis] * b_grad, axis=0)
                bellman_hess = 2 * np.sum(soft_w[:, np.newaxis] * ((self._xi * self._gamma * br[:, np.newaxis] * mm_diag_hess) + b_grad ** 2), axis=0)

            return bellman_grad, bellman_hess

    def compute_gradient_hessian(self, mdp_samples, weights=None):
        if weights is None:
            return super(LinearQMellowBellman, self).compute_gradient_hessian(mdp_samples)
        else:
            r = self._Q.get_statedim() + self._Q.get_actiondim()
            s_prime = self._Q.get_statedim() + self._Q.get_actiondim() + 1
            br = self.bellman_residual(mdp_samples, weights)    # s x w
            mm_gradient, mm_hess = self._gradient_and_hess_mellow_max(mdp_samples[:, s_prime:s_prime + self._Q.get_statedim()], weights=weights)
            q_gradient = self._Q.compute_gradient(mdp_samples[:, 0:r])      # s x f
            b_grad = (self._xi * self._gamma * mm_gradient) - q_gradient[:, :, np.newaxis]  # s x f x w
            if self._tau == 0.0:
                bellman_grad = 2 * np.average(br[:, np.newaxis] * b_grad, axis=0)      # f x w
                bellman_hess = 2 * np.average((self._xi * self._gamma * br[:, np.newaxis, np.newaxis] * mm_hess) \
                                              + b_grad[:, :, np.newaxis] * b_grad[:, np.newaxis], axis=0) # f x f x w
            else:
                soft_w = self._softmax(br ** 2)
                bellman_grad = 2 * np.sum(soft_w[:, np.newaxis] * br[:, np.newaxis] * b_grad, axis=0)  # f x w
                bellman_hess = 2 * np.sum(soft_w[:, np.newaxis, np.newaxis] * ((self._xi * self._gamma * br[:, np.newaxis, np.newaxis] * mm_hess) \
                                              + b_grad[:, :, np.newaxis] * b_grad[:, np.newaxis]), axis=0)  # f x f x w
            return bellman_grad, bellman_hess

    def bellman_residual(self, mdp_samples, weights=None):
        if weights is None:
            return super(LinearQMellowBellman, self).bellman_residual(mdp_samples)
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
            mmQ = np.log(np.sum(mmQ, axis=1)/nacts)/self._kappa + c
            mmQ = mmQ.reshape(Q.shape)

            return mdp_samples[:, r, np.newaxis] + self._gamma * mmQ * (1 - mdp_samples[:, -1, np.newaxis]) - Q

    def _gradient_and_diag_hess_mellow_max(self, states, weights=None):
        if weights is None:
            return super(LinearQMellowBellman, self)._gradient_and_diag_hess_mellow_max(states)
        else:
            q_gradient = self._Q.compute_gradient_all_actions(states) # features

            # Compute normalization constants
            q = np.dot(q_gradient.reshape(q_gradient.shape[0] * q_gradient.shape[1], q_gradient.shape[2]), \
                       weights.T)
            c = np.max(q.reshape(q_gradient.shape[0], q_gradient.shape[1], weights.shape[0]), axis=1)

            qs = self._normalized_mm_exp(q_gradient, c=c, weights=weights)
            qs_sum = np.sum(qs, axis=1)
            qs_sum = qs_sum.reshape(qs_sum.shape[0], 1, qs_sum.shape[1])
            q_gradient = q_gradient[:, :, :, np.newaxis]
            qs = qs[:, :, np.newaxis]
            diag_hess = self._kappa * (np.sum(qs * q_gradient**2, axis=1)/qs_sum \
                                       - (np.sum(qs * q_gradient, axis=1)/qs_sum)**2)
            grad = np.sum(qs * q_gradient, axis=1)/qs_sum
            return grad, diag_hess

    def _gradient_and_hess_mellow_max(self, states, weights=None):
        if weights is None:
            return super(LinearQMellowBellman, self)._gradient_and_hessian_mellow_max(states)
        else:
            q_gradient = self._Q.compute_gradient_all_actions(states) * (1-states[:, -1])[:, np.newaxis, np.newaxis] # features  s x a x f

            # Compute normalization constants
            q = np.dot(q_gradient.reshape(q_gradient.shape[0] * q_gradient.shape[1], q_gradient.shape[2]), \
                       weights.T)  # (s x a) x w
            c = np.max(q.reshape(q_gradient.shape[0], q_gradient.shape[1], weights.shape[0]), axis=1)  # s x w

            qs = self._normalized_mm_exp(q_gradient, c=c, weights=weights)      # s x a x w
            qs_sum = np.sum(qs, axis=1) # s x w

            grad = np.sum(qs[:, :, np.newaxis] * q_gradient[:, :, :, np.newaxis], axis=1)/qs_sum[:, np.newaxis]  # s x f x w

            outer_mm_grad = grad[:,:, np.newaxis] * grad[:, np.newaxis] # s x f x f x w
            outer_q_grad = q_gradient[:, :, :, np.newaxis] * q_gradient[:, :, np.newaxis]  # s x a x f x f

            hess = self._kappa * np.sum(qs[:, :, np.newaxis, np.newaxis] * outer_q_grad[:, :, :, :, np.newaxis], axis=1)/qs_sum[:, np.newaxis, np.newaxis] \
                                       - outer_mm_grad  # s x f x f x w

            return grad, hess

    def _normalized_mm_exp(self, q_values, c=0, weights=None):
        if weights is None:
            return super(LinearQMellowBellman, self)._normalized_mm_exp(q_values, c)
        else:
            # q_values are only features
            nacts = self._Q.actions.size
            qs = q_values.reshape(q_values.shape[0] * q_values.shape[1], q_values.shape[2])
            q = np.dot(qs, weights.T)
            qs = np.exp(self._kappa * (q - np.repeat(c, q_values.shape[1], axis=0)))
            return qs.reshape(q_values.shape[0], q_values.shape[1], weights.shape[0])

    def _softmax(self, X):
       if X.ndim == 1:
           return super(LinearQMellowBellman, self)._softmax(X)
       elif X.ndim == 2:
           # samples x weights
           mx = np.max(X, axis=0)
           num = np.exp(self._tau * (X - mx[np.newaxis]))
           return num / np.sum(num, axis=0)


if __name__ == "__main__":
    import VariationalTransfer.LinearQRegressor as linq
    import features.agrbf as rbf

    gw_size = 5
    n_actions = 2
    state_dim = 2
    action_dim = 1
    n_basis = 2
    K = n_basis ** 2 * n_actions

    x = np.linspace(0, gw_size, n_basis)
    y = np.linspace(0, gw_size, n_basis)
    a = np.linspace(0, n_actions - 1, n_actions)
    mean_x, mean_y, mean_a = np.meshgrid(x, y, a)
    mean = np.hstack((mean_x.reshape(K, 1), mean_y.reshape(K, 1), mean_a.reshape(K, 1)))

    state_var = (gw_size / (n_basis - 1) / 3) ** 2
    action_var = 0.1 ** 2
    covar = np.eye(state_dim + action_dim)
    covar[0:state_dim, 0:state_dim] *= state_var
    covar[-1, -1] *= action_var
    assert covar.shape == (3, 3)
    covar = np.tile(covar, (K, 1))
    assert covar.shape == (3 * K, 3)

    # features
    features = rbf.AGaussianRBF(mean, covar, K=K, dims=state_dim + action_dim)

    q = linq.LinearQRegressor(features, np.arange(n_actions), state_dim, action_dim)
    bellman = LinearQMellowBellman(q, gamma=1, kappa=10, xi=0.0)
    weights = np.ones((2,K))
    weights[1, :] *= 2
    data = np.ones((5, 7))
    statex = np.arange(1,6)
    statey = np.flip(np.arange(1,6), axis=0)
    data[:, -1] *= 0
    data[:, -1] += 1
    data[:, 0] *= statex
    data[:, 1] *= statey
    data[:, 4] *= statey
    data[:, 5] *= statex

    print(bellman.bellman_residual(data))
    print(bellman.compute_gradient_hessian(data, weights))