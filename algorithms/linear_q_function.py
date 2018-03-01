import numpy as np
import features.grbf as rbf
import algorithms.regularized_lsvi as lsvi

class LinearQFunction:

    def __init__(self, actions, features, params=None, gamma=0.99):
        """
        :param actions: tuple with the possible actions available
        :param features: feature object
        :param gamma: discount factor
        :param params: initial parameters

        """
        self._actions = actions
        self._features = features
        self._gamma = gamma
        self._w = params


    def compute_bellman_target(self, samples):
        """
        Computes target as a sampled application of the Bellman optimal operator

        :param samples: samples of the form (s,a,r,s'), one per row (Nx4)
        :return:
        """
        t = list()
        for i in range(samples.shape[0]):
            qs = list()
            for a in self._actions:
                qs.append(np.dot(self._w, self._features(np.asarray((samples[i,0], a)))))
            t.append(samples[i, 3] + self._gamma * (max(qs)))
        return np.array(t)

    def __call__(self, state, action):
        return np.dot(self._features(np.asarray((state, action), dtype=np.float64)), self._w)


if __name__ == '__main__':

    actions = range(4)
    weights = [1, 1]

    mean = np.array([[1, 2], [3, 0]])
    var = np.array([3, 8])

    f = rbf.GaussianRBF(mean, var)

    q = LinearQFunction(actions, f, weights)

    samples = np.random.random((100, 4))
    for k in range(samples.shape[0]):
        samples[k, 1] = np.random.randint(0, 5)
        samples[k, 2] = np.random.randint(-10, 10)

    targets = q.compute_bellman_target(samples)

    print(lsvi.RegularizedLSVI.solve(f(samples[:, 0:2]), np.array((0, 0)), np.eye(2), targets))

    print(q.compute_bellman_target(samples))