import numpy as np


class Policy:
    """Base class for all policies"""

    def sample_action(self, s):
        """Samples from \pi(a|s)"""
        pass


class EpsilonGreedy(Policy):
    """An epsilon-greedy policy"""

    def __init__(self, Q, actions, epsilon=0):
        self._q = Q
        self._e = epsilon
        self._actions = actions

    def sample_action(self, s):
        q_a = self._q.value_actions(s)
        max_a = np.argmax(q_a)

        t = np.random.rand(1)
        if t < 1-self._e:
            return self._actions[max_a]
        else:
            return self._actions[np.random.randint(0, len(self._actions))]


class ScheduledEpsilonGreedy(Policy):
    """An epsilon-greedy policy with scheduled epsilon"""

    def __init__(self, Q, actions, schedule):
        self._q = Q
        self._schedule = schedule
        self._actions = actions
        self._h = 0
        self._H = schedule.shape[0]

    def sample_action(self, s):
        q_a = self._q.value_actions(s)
        max_a = np.argmax(q_a)

        t = np.random.rand(1)
        eps = self._schedule[self._h] if self._h < self._H else self._schedule[-1]
        self._h += 1
        if t < 1-eps:
            return self._actions[max_a]
        else:
            return self._actions[np.random.randint(0, len(self._actions))]