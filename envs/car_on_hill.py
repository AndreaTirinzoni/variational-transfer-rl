import gym
import numpy as np
from gym import spaces
from scipy.integrate import odeint


class CarOnHill(gym.Env):
    """
    The Car On Hill environment as presented in:
    "Tree-Based Batch Mode Reinforcement Learning, D. Ernst et. al."

    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    def __init__(self, mass=1.0, power=4.0):
        self.horizon = 100
        self.gamma = 0.95

        self.max_pos = 1.
        self.max_velocity = 3.

        self._g = 9.81
        self._m = mass
        self._power = power
        self._dt = .1

        # gym attributes
        self.viewer = None
        high = np.array([self.max_pos, self.max_velocity])
        self.observation_space = spaces.Box(low=-high, high=high)
        self.action_space = spaces.Discrete(2)

        # initialize state
        self.seed()
        self.reset()

    def step(self, u):

        u = self._power if u == 0 else -self._power
        sa = np.append(self._state, u)
        new_state = odeint(self._dpds, sa, [0, self._dt])

        self._state = new_state[-1, :-1]

        if self._state[0] < -self.max_pos or \
                np.abs(self._state[1]) > self.max_velocity:
            self._absorbing = True
            reward = -1
        elif self._state[0] > self.max_pos and \
                np.abs(self._state[1]) <= self.max_velocity:
            self._absorbing = True
            reward = 1
        else:
            reward = 0

        return self.get_state(), reward, self._absorbing, {}

    def reset(self, state=None):
        self._absorbing = False
        if state is None:
            pos = np.random.uniform(-0.8,0.8)
            vel = np.random.uniform(-0.5,0.5)
            self._state = np.array([-0.5, 0])
        else:
            self._state = state

        return self.get_state()

    def get_state(self):
        return self._state

    def _dpds(self, state_action, t):
        position = state_action[0]
        velocity = state_action[1]
        u = state_action[-1]

        if position < 0.:
            diff_hill = 2 * position + 1
            diff_2_hill = 2
        else:
            diff_hill = 1 / ((1 + 5 * position ** 2) ** 1.5)
            diff_2_hill = (-15 * position) / ((1 + 5 * position ** 2) ** 2.5)

        dp = velocity
        ds = (u - self._g * self._m * diff_hill - velocity ** 2 * self._m *
              diff_hill * diff_2_hill) / (self._m * (1 + diff_hill ** 2))

        return dp, ds, 0.

    def _render(self, mode=None, close=None):
        pass