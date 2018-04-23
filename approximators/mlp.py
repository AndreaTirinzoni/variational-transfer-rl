#import numpy as np
import autograd.numpy as np
from autograd import jacobian
from approximators.qfunction import QFunction


class MLPQFunction(QFunction):
    """
    A Multi Layer Perceptron Q-function
    """

    def __init__(self, state_dim, n_actions, layers=(32,), initial_params=None):

        self._nn = MLP(state_dim, n_actions, layers)
        self._w = np.random.randn(self._nn.num_weights) * 0.1 if initial_params is None else initial_params
        self._state_dim = state_dim
        self._n_actions = n_actions

    def update_weights(self, w):
        """Updates the regressor's weights"""
        assert w.shape == self._w.shape
        self._w = w

    def value(self, sa):
        """Computes Q(s,a) at each sa [N]"""
        vals = self._nn.predict(sa[:, :-1], self._w[np.newaxis, :])
        assert vals.shape == (1,sa.shape[0],self._n_actions)
        acts = np.array(sa[:, -1], dtype=np.int32)
        assert acts.shape == (sa.shape[0],)
        ret = vals[0, np.arange(sa.shape[0]), acts]
        assert ret.shape == (sa.shape[0],)
        return ret

    def value_actions(self, states, done=None):
        """Computes Q(s,a) for all actions at each s [NxA]"""
        states = states if states.ndim > 1 else states[np.newaxis, :]
        vals = self._nn.predict(states, self._w[np.newaxis, :])
        assert vals.shape == (1,states.shape[0],self._n_actions)
        return vals[0,:,:] * (1 if done is None else 1 - done[:, np.newaxis])

    def gradient(self, sa):
        """Computes the gradient at each sa [NxK]"""
        grads = self._nn.gradient(sa[:, :-1], self._w[np.newaxis, :])
        assert grads.shape == (1,sa.shape[0],self._n_actions,self._w.shape[0])
        acts = np.array(sa[:, -1], dtype=np.int32)
        assert acts.shape == (sa.shape[0],)
        ret = grads[0, np.arange(sa.shape[0]), acts, :]
        assert ret.shape == (sa.shape[0],self._w.shape[0])
        return ret

    def gradient_actions(self, states):
        """Computes the gradient for all actions at each s [NxAxK]"""
        grads = self._nn.gradient(states, self._w[np.newaxis, :])
        assert grads.shape == (1,states.shape[0],self._n_actions,self._w.shape[0])
        return grads[0,:,:,:]

    def value_gradient(self, sa):
        """Computes Q(s,a) and its gradient at each sa"""
        vals = self.value(sa)
        grads = self.gradient(sa)
        return vals, grads

    def value_gradient_actions(self, states, done=None):
        """Computes Q(s,a) and its gradient for all actions at each s"""
        vals = self.value_actions(states, done)
        grads = self.gradient_actions(states) * (1 if done is None else 1 - done[:, np.newaxis, np.newaxis])
        return vals, grads

    def value_weights(self, sa, weights):
        """Computes Q(s,a) for any weight passed at each sa [NxM]"""
        vals = self._nn.predict(sa[:, :-1], weights)
        assert vals.shape == (weights.shape[0], sa.shape[0], self._n_actions)
        acts = np.array(sa[:, -1], dtype=np.int32)
        assert acts.shape == (sa.shape[0],)
        ret = vals[:, np.arange(sa.shape[0]), acts]
        assert ret.shape == (weights.shape[0], sa.shape[0])
        return ret.T

    def value_actions_weights(self, states, weights, done=None):
        """Computes Q(s,a) for any action and for any weight passed at each s [NxAxM]"""
        vals = self._nn.predict(states, weights)
        assert vals.shape == (weights.shape[0],states.shape[0],self._n_actions)
        return np.transpose(vals, (1,2,0)) * (1 if done is None else 1 - done[:, np.newaxis, np.newaxis])

    def gradient_weights(self, sa, weights):
        """Computes the gradient for each weight at each sa [NxKxM]"""
        grads = np.zeros((weights.shape[0],sa.shape[0],self._n_actions,self._w.shape[0]))
        for i in range(weights.shape[0]):
            grads[i,:,:,:] = self._nn.gradient(sa[:, :-1], weights[i,:].reshape(1,weights.shape[1]))[0,:,:,:]
        # TODO : it seems that a for loop is still faster
        #grads = self._nn.gradient(sa[:, :-1], weights)
        assert grads.shape == (weights.shape[0],sa.shape[0],self._n_actions,self._w.shape[0])
        acts = np.array(sa[:, -1], dtype=np.int32)
        assert acts.shape == (sa.shape[0],)
        ret = grads[:, np.arange(sa.shape[0]), acts, :]
        assert ret.shape == (weights.shape[0],sa.shape[0],self._w.shape[0])
        return np.transpose(ret, (1,2,0))

    def gradient_actions_weights(self, states, weights):
        """Computes the gradient for all actions and weights at each s [NxAxKxM]"""
        grads = np.zeros((weights.shape[0],states.shape[0],self._n_actions,self._w.shape[0]))
        for i in range(weights.shape[0]):
            grads[i,:,:,:] = self._nn.gradient(states, weights[i,:].reshape(1,weights.shape[1]))[0,:,:,:]
        # TODO : it seems that a for loop is still faster
        #grads = self._nn.gradient(states, weights)
        assert grads.shape == (weights.shape[0],states.shape[0],self._n_actions,self._w.shape[0])
        return np.transpose(grads, (1,2,3,0))

    def value_gradient_weights(self, sa, weights):
        """Computes Q(s,a) and its gradient at each sa"""
        vals = self.gradient_weights(sa, weights)
        grads = self.gradient_weights(sa, weights)
        return vals, grads

    def value_gradient_actions_weights(self, states, weights, done=None):
        """Computes Q(s,a) and its gradient for all actions at each s"""
        vals = self.value_actions_weights(states, weights, done)
        grads = self.gradient_actions_weights(states, weights) * (1 if done is None else 1 - done[:, np.newaxis, np.newaxis, np.newaxis])
        return vals, grads

class MLP:
    """A neural network"""

    relu = lambda x: np.maximum(x, 0.)

    def __init__(self, state_dim, n_actions, layers, activation=np.tanh):

        layers = [state_dim] + list(layers)
        layers.append(n_actions)
        self._shapes = list(zip(layers[:-1], layers[1:]))
        self.num_weights = sum((m+1)*n for m, n in self._shapes)
        self._activation = activation

    def _unpack_layers(self, weights):
        """Unpacks a set of weights (MxK) into the corresponding layers"""
        nw = len(weights)
        for m, n in self._shapes:
            yield weights[:, :m*n].reshape((nw, m, n)), weights[:, m*n:m*n+n].reshape((nw, 1, n))
            weights = weights[:, (m+1)*n:]

    def predict(self, inputs, weights):
        """Computes the output (MxNxA) for different inputs (NxS) and weights (MxK)"""
        inputs = np.expand_dims(inputs, 0)
        for W, b in self._unpack_layers(weights):
            outputs = np.einsum('mnd,mdo->mno', inputs, W) + b
            inputs = self._activation(outputs)
        return outputs

    def gradient(self, inputs, weights):
        """Computes the gradient (MxNxAxK) for different inputs (NxS) and weights (MxK)"""
        f = lambda w : self.predict(inputs, w)
        g = jacobian(f)(weights)
        v = np.arange(g.shape[0])
        return g[v,:,:,v,:]