import numpy as np


class QFunction:
    """
    Base interface for Q functions
    """

    def update_weights(self, w):
        """Updates the regressor's weights"""
        pass

    def value(self, sa):
        """Computes Q(s,a) at each sa"""
        pass

    def value_actions(self, states, done=None):
        """Computes Q(s,a) for all actions at each s"""
        pass

    def gradient(self, sa):
        """Computes the gradient at each sa"""
        pass

    def gradient_actions(self, states):
        """Computes the gradient for all actions at each s"""
        pass

    def value_gradient(self, sa):
        """Computes Q(s,a) and its gradient at each sa"""
        pass

    def value_gradient_actions(self, states, done=None):
        """Computes Q(s,a) and its gradient for all actions at each s"""
        pass

    def value_weights(self, sa, weights):
        """Computes Q(s,a) for any weight passed at each sa"""
        pass

    def value_actions_weights(self, states, weights, done=None):
        """Computes Q(s,a) for any action and for any weight passed at each s"""
        pass

    def gradient_weights(self, sa, weights):
        """Computes the gradient for each weight at each sa"""
        pass

    def gradient_actions_weights(self, states, weights):
        """Computes the gradient for all actions and weights at each s"""
        pass

    def value_gradient_weights(self, sa, weights):
        """Computes Q(s,a) and its gradient at each sa"""
        pass

    def value_gradient_actions_weights(self, states, weights, done=None):
        """Computes Q(s,a) and its gradient for all actions at each s"""
        pass
