#import numpy as np
from approximators.qfunction import QFunction

import torch
from torch.nn import ReLU
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter
import numpy as np


class MLPQFunction(QFunction):
    """
    A Multi Layer Perceptron Q-function
    """
    def __init__(self, state_dim, n_actions, layers=(32,), initial_params=None):
        self._nn = Net(state_dim, n_actions, layers)
        self._w = np.random.randn(self._nn.n_weights) * 0.1 if initial_params is None else initial_params
        self._state_dim = state_dim
        self._n_actions = n_actions

    def update_weights(self, w):
        """Updates the regressor's weights"""
        if w.ndim == 1:
            assert w.size == self._nn.n_weights
        else:
            assert w.ndim == 2 and w.shape[1] == self._nn.n_weights
        self._w = w
        self._nn.set_weights(w)

    def value(self, sa, grad_required=False):
        """Computes Q(s,a) at each sa [N]"""
        vals = self._nn.forward(sa[:, :-1])    # tensor
        assert vals.shape == (sa.shape[0], self._n_actions)
        acts = np.array(sa[:, -1], dtype=np.int32)
        assert acts.shape == (sa.shape[0],)
        ret = vals[np.arange(sa.shape[0]), acts]
        assert ret.shape == (sa.shape[0],)
        return ret.detach().numpy() if not grad_required else ret

    def value_actions(self, states, done=None, grad_required=False):
        """Computes Q(s,a) for all actions at each s [NxA]"""

        states = states if states.ndim > 1 else states[np.newaxis, :]
        vals = self._nn.forward(states)    # tensor
        assert vals.shape == (states.shape[0], self._n_actions)
        return vals * (1 if done is None else 1 - torch.from_numpy(done[:, np.newaxis])) if grad_required \
                else vals.detach().numpy() * (1 if done is None else 1 - done[:, np.newaxis])

    def value_weights(self, sa, weights=None, grad_required=False):
        """Computes Q(s,a) for any weight passed at each sa [NxM]"""
        if weights is not None:
            self.update_weights(weights)
        vals = self._nn.forward(sa[:, :-1])
        acts = torch.from_numpy(sa[:, -1]).type(torch.int64)
        ret = vals[:, torch.arange(sa.shape[0], dtype=torch.int64), acts] * 1
        return torch.einsum("sw->ws", (ret,)) if grad_required else ret.detach().numpy().T

    def value_actions_weights(self, states, weights=None, done=None, grad_required=False):
        """Computes Q(s,a) for any action and for any weight passed at each s [NxAxM]"""
        if weights is not None:
            self.update_weights(weights)
        vals = self._nn.forward(states)
        return np.transpose(vals.detach().numpy(), (1, 2, 0)) * (1 if done is None else 1 - done[:, np.newaxis, np.newaxis]) if not grad_required \
                else torch.einsum("aws->wsa", (vals,)) * (1 if done is None else 1 - torch.from_numpy(done[:, np.newaxis, np.newaxis]))

    def gradient(self, prepare=False):
        torch.autograd.enable_grad()
        if prepare:
            self._nn.zero_grad() # prepare for gradient
        else:
            return self._nn.grad()

class Net(nn.Module):
    """A neural network"""

    def __init__(self, state_dim, n_actions, layers):
        super(Net, self).__init__()
        self.double()
        self.input_layer = BatchedWeightedLinear(state_dim, layers[0])
        self.hidden_layers = []
        for l in range(len(layers)-1):
            self.hidden_layers.append(BatchedWeightedLinear(layers[l],layers[l+1]))
            self.add_module("hidden_layer" + str(l), self.hidden_layers[-1])
        self.output_layer = BatchedWeightedLinear(layers[-1], n_actions)

        self._shapes = [list(p.size()) for p in list(self.parameters())]
        self._weights = [p.data.numpy() for p in list(self.parameters())]
        self._sizes = [p.size for p in self._weights]
        self._indexes = np.cumsum(self._sizes)[:-1]
        self.n_weights = np.sum(self._sizes)
        self.weight_batch = 1

    def get_weights(self):
        return np.concatenate([w.flatten() for w in self._weights])

    def set_weights(self, w):
        if w.ndim == 1:
            assert w.size == self.n_weights
            if self.weight_batch != 1:
                self.set_weight_batch(1)
            for ow,shape,nw in zip(self._weights, self._shapes, np.split(w, self._indexes)):
                ow[:] = nw.reshape(shape)
        elif w.ndim == 2:
            assert w.shape[1] == self.n_weights
            if w.shape[1] != self.weight_batch:
                self.set_weight_batch(w.shape[0])
            for ow,shape,nw in zip(self._weights, self._shapes, np.split(w, self._indexes, axis=1)):
                ow[:,:] = nw.reshape(shape)
        self.zero_grad()

    def set_weight_batch(self, size):
        assert size > 0
        if size != self.weight_batch:
            self.input_layer.set_batch_size(size)
            self.output_layer.set_batch_size(size)
            for l in self.hidden_layers:
                l.set_batch_size(size)
            self._shapes = [list(p.size()) for p in list(self.parameters())]
            self._weights = [p.data.numpy() for p in list(self.parameters())]
            self.weight_batch = size

    def forward(self, x):
        x = torch.from_numpy(x)
        if self.weight_batch > 1:
            x = x.unsqueeze(0)
        x = F.relu(self.input_layer(x))
        for l in self.hidden_layers:
            x = F.relu(l(x))
        x = self.output_layer(x)
        return x

    def grad(self):
        g = [p.grad.data.numpy() for p in self.parameters()]
        if self.weight_batch > 1:
            g = np.concatenate([w.reshape(self.weight_batch, size) for (w,size) in zip(g,self._sizes)])
        else:
            g = np.concatenate([w.flatten() for w in g])
        return g

class BatchedWeightedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super(BatchedWeightedLinear, self).__init__(in_features, out_features)
        self.weight_batch = 1 # size of the current batch for the weights
        self.double()

    def forward(self, x):
        if self.weight.data.dim() == 2:
            return super(BatchedWeightedLinear, self).forward(x)
        elif self.weight.data.dim() == 3:
            return torch.einsum("mni,mio->mno", (x, self.weight)) + self.bias.unsqueeze(1)
        else:
            raise Exception("No more than one batch dimension supported for the weights")

    def get_batch_size(self):
        return self.weight_batch

    def set_batch_size(self, size):
        assert size > 0
        self.weight_batch = size
        if size > 1:
            self.weight = Parameter(torch.Tensor(size, self.in_features, self.out_features))
            self.bias = Parameter(torch.Tensor(size, self.out_features))
        else:
            self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))
            self.bias = Parameter(torch.Tensor(self.out_features))
        self.double()


if __name__== "__main__":

    nn = Net(2,1,[32,])
    w = torch.autograd.Variable(torch.randn((10, nn.n_weights)), requires_grad=True)
    samples = np.random.rand(10,2)
    t = torch.rand(10,1)

    nn.set_weights(w.data.numpy())
    loss = ((nn(samples)-t) ** 2).mean()
    w1 = w[0]
    nn.set_weights(w1.data.numpy())
    loss = ((nn(samples) - t) ** 2).mean()

    loss.backward()