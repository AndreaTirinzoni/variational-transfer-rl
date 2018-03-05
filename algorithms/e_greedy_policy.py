import numpy as np

class eGreedyPolicy:

    def __init__(self, Q, actions, epsilon=0):
        self._q = Q
        self._e = epsilon
        self._actions = actions

    def sample_action(self, s):
        q_a = self._q.compute_all_actions(s)
        max_a = np.argmax(q_a)
        t = np.random.rand(1)
        if t < 1-self._e:
            return self._actions[max_a]
        else:
            return self._actions[np.random.randint(0, len(self._actions))]
