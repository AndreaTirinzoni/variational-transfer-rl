import numpy as np
import features.grbf as rbf
import algorithms.regularized_lsvi as lsvi
import utils

class LinearQFunction:

    def __init__(self, actions, features, params=None, state_dim=1, action_dim=1, gamma=0.99):
        """
        :param actions: tuple with the possible actions available
        :param features: feature object
        :param gamma: discount factor
        :param params: initial parameters
        :param state_dim: dimension of the state vector
        :param action_dim: dimension of the action vector

        """
        self.actions = actions
        self._features = features
        self._gamma = gamma
        self._w = [params for i in range(len(actions))]
        self._state_dim = state_dim
        self._action_dim = action_dim


    def compute_bellman_target(self, samples):
        """
        Computes target as a sampled application of the Bellman optimal operator

        :param samples: samples of the form (s,a,r,s',terminating), one per row (Nx5)
        :return:
        """

        s_prime = self._state_dim + self._action_dim + 1
        r = self._state_dim + self._action_dim
        t = list()
        a = self._state_dim
        for i in range(samples.shape[0]):
            qs = list()
            fts = self._features(samples[i, s_prime: s_prime + self._state_dim])
            for a in self.actions:
                qs.append(np.dot(self._w[a], fts))
            t.append((samples[i, r] + self._gamma * max(qs)) if samples[i, -1] == 0 else samples[i, r])
        return np.array(t)


    def update_weights(self, w, action=None):
        """
        Update the weights of the regressors
        :param w: vector of weights or list of vectors of weights
        :param action: specific action to which update the regressor's parameters
        """
        if action is None:
            for i in range(len(w)):
                assert w[i].shape == self._w[i].shape
            self._w = w
        else:
            assert w.shape == self._w[action].shape
            self._w[action] = w

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
    weights = [1, 1]

    mean = np.array([[1, 2], [3, 0]])
    var = np.array([3, 8])

    f = rbf.GaussianRBF(mean, var)

    q = LinearQFunction(actions, f, weights)

    samples = np.random.random((100, 4))
    term = np.random.random_integers(0, 2, (100, 1))

    samples = np.hstack((samples, term))
    for k in range(samples.shape[0]):
        samples[k, 1] = np.random.randint(0, 5)
        samples[k, 2] = np.random.randint(-10, 10)

    targets = q.compute_bellman_target(samples)

    print(lsvi.RegularizedLSVI.solve(f(samples[:, 0:2]), targets, np.array((0, 0)), np.eye(2)))

    print(q.compute_bellman_target(samples))