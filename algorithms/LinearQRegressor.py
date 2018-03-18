import numpy as np

"""
Linear Regressor for Q function
"""

class LinearQRegressor:

    def __init__(self, features, actions, state_dim=1, action_dim=1, initial_params=None):

        if initial_params is not None:
            assert initial_params.size == features.number_of_features()
            self._w = initial_params

        self._actions = actions
        self._features = features
        self.state_dim = state_dim
        self.action_dim = action_dim

    def update_weights(self, weights):
        assert weights.shape == self._w.shape
        self._w = weights

    def compute_all_actions(self, state):
        t = np.hstack((np.tile(state, (self._actions.size, 1)), self._actions))
        f = self._features(t)
        return np.dot(f, self._w).reshape(state.shape[0], self._actions.size)

    def compute_gradient(self, state_action):
        return self._features(state_action)

    def compute_gradient_all_actions(self, state):
        t = np.hstack((np.tile(state, (self._actions.size, 1)), self._actions))
        f = self._features(t)
        return f.reshape(state.shape[0], self._actions.size, f.shape[1])

    def compute_diag_hessian_all_actions(self, state):
        return np.zeros((state.shape[0], self._actions.size, self._features.number_of_features()))

    def __call__(self, state_action):
        return np.dot(self._features(state_action), self._w)