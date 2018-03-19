import numpy as np

"""
Linear Regressor for Q function
"""

class LinearQRegressor:

    def __init__(self, features, actions, state_dim=1, action_dim=1, initial_params=None):

        if initial_params is not None:
            assert initial_params.size == features.number_of_features()
            self._w = initial_params
        else:
            self._w = np.zeros(features.number_of_features())
        self.actions = actions[:, np.newaxis]
        self._features = features
        self._state_dim = state_dim
        self._action_dim = action_dim

    def update_weights(self, weights):
        assert weights.shape == self._w.shape
        self._w = weights

    def compute_all_actions(self, state):

        s = state[:, np.newaxis].T if state.ndim == 1 else state
        n_state = s.shape[0]
        s = np.repeat(s, self.actions.size, axis=0)
        a = np.tile(self.actions, (n_state, 1))
        t = np.hstack((s, a))
        f = self._features(t)
        return np.dot(f, self._w).reshape(n_state, self.actions.size)

    def compute_gradient(self, state_action):
        return self._features(state_action)

    def compute_gradient_all_actions(self, state):
        s = state[:, np.newaxis].T if state.ndim == 1 else state
        nstate = s.shape[0]
        t = np.hstack((np.repeat(s, self.actions.size, axis=0), np.tile(self.actions, (nstate, 1))))
        f = self._features(t)
        return f.reshape(state.shape[0], self.actions.size, f.shape[1])

    def compute_diag_hessian_all_actions(self, state):
        return np.zeros((state.shape[0], self.actions.size, self._features.number_of_features()))

    def __call__(self, state_action):
        return np.dot(self._features(state_action), self._w)

    def compute_features(self, samples):
        return self._features(samples[:, 0: self._state_dim + self._action_dim])

    def get_statedim(self):
        return self._state_dim

    def get_actiondim(self):
        return self._action_dim