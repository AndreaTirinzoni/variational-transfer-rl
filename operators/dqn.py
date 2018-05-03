import utils
from operators.operator import Operator
from approximators.mlp_torch import MLPQFunction
from torch.nn.functional import smooth_l1_loss
import numpy as np
import torch


class DQNOperator(Operator):
    """ Bellman operator based on Huber Loss using Double Q Network trick.
        This keeps track of the gradient calls made to update the target network given a frequency of
        updates """

    def __init__(self, state_dim, action_dim,
                 gamma,
                 target_q,
                 target_update_freq=500):

        self._gamma = gamma
        self._state_dim = state_dim
        self._action_dim = action_dim
        # target network
        self._q_target = target_q
        self._t = 0
        self._target_update_freq = target_update_freq

    def bellman_residual(self, Q, samples, weights=None):
        """General function for computing Bellman residuals"""
        _, _, _, r, s_prime, absorbing, sa = utils.split_data(samples, self._state_dim, self._action_dim)
        return self._bellman_residual_single(Q, r, s_prime, absorbing, sa) if weights is None else \
            self._bellman_residual_multi(Q, r, s_prime, absorbing, sa, weights)

    def _bellman_residual_single(self, Q, r, s_prime, absorbing, sa):
        """Computes the Bellman residuals of given samples"""
        Qs_prime = Q.value_actions(s_prime, absorbing)
        maxQ = self._q_target.value(np.concatenate((s_prime, np.argmax(Qs_prime, axis=1)[:, np.newaxis]), axis=1))
        return r + self._gamma * maxQ * (1 - absorbing) - Q.value(sa)

    def _bellman_residual_multi(self, Q, r, s_prime, absorbing, sa, weights):
        """Computes the Bellman residuals of a set of samples given a set of weights"""
        current_w = Q._w
        Q_values_prime = Q.value_actions_weights(s_prime, weights, absorbing)
        amax = np.argmax(Q_values_prime, axis=1).astype("int64")
        state = np.repeat(np.arange(s_prime.shape[0], dtype="int64"), weights.shape[0])
        maxQ = self._q_target.value_actions(s_prime, absorbing)
        maxQ = maxQ[state, amax.flatten()].reshape(s_prime.shape[0], weights.shape[0])
        ret = r[:, None] + self._gamma * maxQ * (1 - absorbing[:, None]) - Q.value_weights(sa, weights)
        Q._w = current_w
        return ret

    def _bellman_residual_surrogate(self, Q, samples, weights=None):
        _, _, _, r, s_prime, absorbing, sa = utils.split_data(samples, self._state_dim, self._action_dim)
        if weights is None:
            amax = torch.argmax(Q.value_actions(s_prime, absorbing, grad_required=True), dim=1)
            amax = amax.detach().numpy()    # ensure that is not taken for the derivative
            maxQ = self._q_target.value(np.concatenate((s_prime, amax[:,np.newaxis]), axis=1), grad_required=True).detach()
            r = torch.from_numpy(r)
            absorbing = torch.from_numpy(absorbing)
            qval = Q.value(sa, grad_required=True)
        else:
            qprime = Q.value_actions_weights(s_prime, weights=weights, done=absorbing, grad_required=True).detach()
            amax = torch.argmax(qprime, dim=1).type("int64") # best actions
            maxQ = self._q_target.value_actions_weights(s_prime, weights=weights, done=absorbing, grad_required=True)

            state = np.repeat(np.arange(s_prime.shape[0], dtype="int64"), weights.shape[0])
            amax = amax.view(-1)        # flattens the tensor
            maxQ = maxQ[state, amax].view(s_prime.shape[0], weights.shape[0]).detach()
            r = torch.from_numpy(r).unsqueeze(1)
            absorbing = torch.from_numpy(absorbing).unsqueeze(1)
            qval = Q.value_weights(sa, grad_required=True)
        return smooth_l1_loss(qval, r + self._gamma * maxQ * (1-absorbing), reduce=False)

    def gradient_be(self, Q, samples, weights=None):
        """General function for gradients of the Bellman error"""
        with torch.enable_grad():
            Q.gradient(prepare=True)    # zeroes out old grads.
            if weights is not None:
                current_w = Q._w
                g = []
                for w in range(weights.shape[0]):       # TODO this for loop seems slightly faster
                    Q.gradient(prepare=True)
                    Q._w = weights[w,:]
                    be = self._bellman_residual_surrogate(Q, samples).mean()
                    be.backward()
                    g.append(Q.gradient())

                Q._w = current_w
                grad = np.array(g)
            else:
                be = self._bellman_residual_surrogate(Q, samples).mean()
                be.backward()
                grad = Q.gradient()

        ## Every gradient step is assumed to be a timestep.
        ## implementing it this way updates to the network just before taking the grad step
        self._t += 1
        if self._t % self._target_update_freq == 0:
            self._q_target._w = Q._w
        return grad

    def bellman_error(self, Q, samples, weights=None):
        """General function for computing the Bellman error"""
        return self._bellman_error_single(Q, samples) if weights is None else \
            self._bellman_error_multi(Q, samples, weights)

    def _bellman_error_single(self, Q, samples):
        """Computes the Bellman error"""
        br = self.bellman_residual(Q, samples) ** 2
        return br.sum()

    def _bellman_error_multi(self, Q, samples, weights):
        """Computes the Bellman error for each weight"""
        br = self.bellman_residual(Q, samples, weights) ** 2
        errors = np.average(br, axis=0)
        return errors

    def expected_bellman_error(self, Q, samples, weights):
        """Approximates the expected Bellman error with a finite sample of weights"""
        return np.average(self.bellman_error(Q, samples, weights))


if __name__ == "__main__":
    l1 = 5
    l2 = 0
    kappa = 100.
    tau = 0.
    xi = 0.5
    gamma = 0.99
    state_dim = 2
    action_dim = 1
    n_actions = 10

    layers = [l1]
    if l2 > 0:
        layers.append(l2)

    samples = np.random.randn(10, 1 + state_dim + action_dim + 1 + state_dim + 1)
    samples[:, -1] = 0.
    samples[:, action_dim + state_dim] = np.random.random_integers(0, n_actions - 1, size=samples.shape[0])


    Q, operator = DQN(state_dim, action_dim, n_actions, gamma, layers=layers)
    weights = np.random.randn(5, Q._w.shape[0])

    br = operator.bellman_residual(Q, samples,weights)
    g = operator.gradient_be(Q,samples,weights)

    h = 1
