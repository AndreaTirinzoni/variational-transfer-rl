import numpy as np
from approximators.qfunction import QFunction


class LinearQFunction(QFunction):
    """
    Linear Regressor for Q function
    """

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

    def update_weights(self, w):
        """Updates the regressor's weights"""
        assert w.shape == self._w.shape
        self._w = w

    def value(self, sa):
        """Computes Q(s,a) at each sa"""
        return np.dot(self._features(sa), self._w)

    def value_actions(self, states, done=None):
        """Computes Q(s,a) for all actions at each s"""
        feats = self.gradient_actions(states)
        return np.dot(feats, self._w) * (1 if done is None else 1 - done[:, np.newaxis])

    def gradient(self, sa):
        """Computes the gradient at each sa"""
        return self._features(sa)

    def gradient_actions(self, states):
        """Computes the gradient for all actions at each s"""
        s = states[:, np.newaxis].T if states.ndim == 1 else states
        nstate = s.shape[0]
        t = np.hstack((np.repeat(s, self.actions.size, axis=0), np.tile(self.actions, (nstate, 1))))
        f = self._features(t)
        return f.reshape(s.shape[0], self.actions.size, f.shape[1])

    def value_gradient(self, sa):
        """Computes Q(s,a) and its gradient at each sa"""
        feats = self._features(sa)
        return np.dot(feats, self._w), feats

    def value_gradient_actions(self, states, done=None):
        """Computes Q(s,a) and its gradient for all actions at each s"""
        # We zero-out features corresponding to terminal states so that their value and their gradient are zero
        feats = self.gradient_actions(states) * (1 if done is None else 1 - done[:, np.newaxis, np.newaxis])
        return np.dot(feats, self._w), feats

    def value_weights(self, sa, weights):
        """Computes Q(s,a) for any weight passed at each sa"""
        feats = self.gradient(sa)
        return np.dot(feats, weights.T)

    def value_actions_weights(self, states, weights, done=None):
        """Computes Q(s,a) for any action and for any weight passed at each s"""
        feats = self.gradient_actions(states)
        return np.dot(feats, weights.T) * (1 if done is None else 1 - done[:, np.newaxis, np.newaxis])

    def gradient_weights(self, sa, weights):
        """Computes the gradient for each weight at each sa"""
        return self.gradient(sa)

    def gradient_actions_weights(self, states, weights):
        """Computes the gradient for all actions and weights at each s"""
        return self.gradient_actions(states)

    def value_gradient_weights(self, sa, weights):
        """Computes Q(s,a) and its gradient at each sa"""
        feats = self._features(sa)
        return np.dot(feats, weights.T), feats

    def value_gradient_actions_weights(self, states, weights, done=None):
        """Computes Q(s,a) and its gradient for all actions at each s"""
        # We zero-out features corresponding to terminal states so that their value and their gradient are zero
        feats = self.gradient_actions(states) * (1 if done is None else 1 - done[:, np.newaxis, np.newaxis])
        return np.dot(feats, weights.T), feats
