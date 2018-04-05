import numpy as np

class expectedPolicy:

    def __init__(self, Q, actions, posterior):
        self._q = Q
        self._posterior = posterior
        self._actions = actions

    def sample_action(self, s):
        self._q.update_weights(self._posterior.sample(1)[0])
        q_a = self._q.compute_all_actions(s)
        max_a = np.argmax(q_a)
        return self._actions[max_a]