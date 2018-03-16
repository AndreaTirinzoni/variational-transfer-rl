import numpy as np
import features.grbf as rbf
import algorithms.regularized_lsvi as lsvi
import utils

class LinearQFunction:

    def __init__(self, actions, features, params=None, state_dim=1, action_dim=1, gamma=0.99, regressor_per_action=True):

        """
        :param actions: tuple with the possible actions available
        :param features: feature object
        :param gamma: discount factor
        :param params: initial parameters (np.ndarray n_features x n_actions or n_features x 1)
        :param state_dim: dimension of the state vector
        :param action_dim: dimension of the action vector
        :param regressor_per_action: boolean flag to have or not a regressor per action available

        """
        if params is not None:
            if regressor_per_action:
                assert params.shape == (features.number_of_features(), len(actions))
            else:
                assert params.shape == (features.number_of_features(),)

        else:
            if regressor_per_action:
                self._w = np.zeros((features.number_of_features(), len(actions)))
            else:
                self._w = np.zeros((features.number_of_features(),))

        self.actions = actions
        self._features = features
        self._gamma = gamma
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._isRxA = regressor_per_action


    def compute_bellman_target(self, samples):
        """
        Computes target as a sampled application of the Bellman optimal operator

        :param samples: samples of the form (s,a,r,s',terminating), one per row (Nx5)
        :return:
        """

        s_prime = self._state_dim + self._action_dim + 1
        r = self._state_dim + self._action_dim
        a = self._state_dim
        fts = self._features(samples[:, s_prime: s_prime + self._state_dim])
        qs = np.max(np.dot(fts, self._w), axis=1)
        t = samples[:, r] + self._gamma * qs * (1-samples[:, -1])
        return t


    def update_weights(self, w, action=None):
        """
        Update the weights of the regressors
        :param w: matrix of weights (one regressor per column)
        :param action: specific action to which update the regressor's parameters
        """
        if action is None:
            assert w.shape == self._w.shape
            self._w = w
        else:
            assert w.shape == self._w[:, action].shape
            self._w[:, action] = w

    def compute_all_actions(self, state):
        q = list()
        for a in self.actions:
            q.append(self.__call__(state, a))
        return np.array(q)

    def compute_features(self, samples):
        return self._features(samples[:, 0: self._state_dim])

    def __call__(self, state, action):
        return np.dot(self._w[action], self._features(state))


if __name__ == '__main__':

    actions = range(4)
    weights = np.zeros((2, 4))

    mean = np.array([[1, 2], [3, 0]])
    var = np.array([3, 8])

    f = rbf.GaussianRBF(mean, var)

    q = LinearQFunction(actions, f, weights, state_dim=2, action_dim=1)

    samples = np.random.random((100, 5))
    term = np.random.random_integers(0, 2, (100, 1))

    samples = np.hstack((samples, term))
    samples[:, 2] = np.random.randint(0, 5, (samples.shape[0]))


    targets = q.compute_bellman_target(samples)

    print(lsvi.RegularizedLSVI.solve(f(samples[:, 0:2]), targets, np.array((0, 0)), np.eye(2)))

    print(q.compute_bellman_target(samples))