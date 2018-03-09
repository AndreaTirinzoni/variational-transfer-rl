import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import Adam
import tensorflow as tf


def get_regularizer(mu, sigma_inv, reg_lambda):
    if mu is None or sigma_inv is None:
        return None

    assert mu.shape[0] == sigma_inv.shape[0] and mu.shape[0] == sigma_inv.shape[1]

    mu_tf = tf.constant(mu, dtype=tf.float32)
    sigma_inv_tf = tf.constant(sigma_inv, dtype=tf.float32)

    def reg(weight_matrix):
        diff = tf.subtract(tf.reshape(weight_matrix, [-1]), mu_tf)
        return reg_lambda * tf.matmul(tf.matmul(tf.expand_dims(diff,0), sigma_inv_tf), tf.expand_dims(diff,-1))
    return reg


class NNQ:

    def __init__(self, actions, state_dim, gamma=0.99, layers=(32,32), activation="relu", prior_mean=None, prior_cov=None, reg_lambda = 1.0):
        """
        :param actions: list of discrete actions
        :param state_dim: the state dimension (scalar > 0)
        :param gamma: discount factor
        :param layers: tuple containing the number of neurons in each layer
        :param activation: activation functions
        :param prior_mean: mean of the Gaussian prior
        :param prior_cov: covariance matrix of the Gaussian prior
        """
        self.actions = actions
        self.n_actions = len(actions)
        # Binarize the actions
        tmp = np.eye(self.n_actions)
        self.bin_actions = [tmp[i,:] for i in range(self.n_actions)]
        self._state_dim = state_dim
        self._gamma = gamma

        # Compute the size of each weight vector
        sizes = []
        sizes.append((state_dim + self.n_actions) * layers[0])
        sizes.append(layers[0])
        for l in range(len(layers) - 1):
            sizes.append(layers[l] * layers[l + 1])
            sizes.append(layers[l + 1])
        sizes.append(layers[-1])
        sizes.append(1)

        if prior_mean is not None and prior_cov is not None:
            # Split the prior for different layers
            means = np.split(prior_mean, np.cumsum(sizes)[:-1])
            prior_cov = np.diag(prior_cov)
            covs = np.split(prior_cov, np.cumsum(sizes)[:-1])
            inv_covs = [np.linalg.inv(np.diag(c)) for c in covs]
        else:
            means = [None for _ in sizes]
            inv_covs = [None for _ in sizes]

        # Build the NN
        self._nn = Sequential()
        self._nn.add(Dense(layers[0], input_dim=state_dim+self.n_actions,
                           kernel_regularizer=get_regularizer(means[0],inv_covs[0],reg_lambda),
                           bias_regularizer=get_regularizer(means[1],inv_covs[1],reg_lambda)))
        self._nn.add(Activation(activation))
        idx = 2
        for l in range(len(layers)-1):
            self._nn.add(Dense(layers[l+1], kernel_regularizer=get_regularizer(means[idx],inv_covs[idx],reg_lambda),
                               bias_regularizer=get_regularizer(means[idx+1],inv_covs[idx+1],reg_lambda)))
            self._nn.add(Activation(activation))
            idx += 2
        self._nn.add(Dense(1, kernel_regularizer=get_regularizer(means[idx],inv_covs[idx],reg_lambda),
                           bias_regularizer=get_regularizer(means[idx+1],inv_covs[idx+1],reg_lambda)))
        self._nn.add(Activation('linear'))
        self._nn.compile(loss='mean_squared_error', optimizer=Adam())

        self._fitted = False

        self._prior_mean = prior_mean
        self._prior_cov = prior_cov

        # TODO: Should we allow sampling from the prior?
        if prior_mean is not None:
            self.update_weights(prior_mean)

    def compute_bellman_target(self, r, s_prime, absorbing):
        """
        Computes the Bellman target for all samples
        :param r: Nx1 array of rewards
        :param s_prime: NxS array of states
        :param absorbing: Nx1 array specifying whether s_i is a terminal states
        :return: Nx1 array with the targets
        """
        y = np.array(r)
        if self._fitted:
            for i in range(r.shape[0]):
                if not absorbing[i]:
                    y[i] += self._gamma * self._nn.predict(np.array([np.concatenate((s_prime[i,:],a)) for a in self.bin_actions])).max()
        return y

    def compute_all_actions(self, state):
        """
        Computes all action values for the given state
        :param state: S-dimensional array
        :return: A-dimensional array
        """
        return self._nn.predict(np.array([np.concatenate((state,a)) for a in self.bin_actions]))

    def __call__(self, state, action):
        """
        Computes the value of a pair (s,a)
        :param state: S-dimensional array
        :param action: scalar index in [0,A)
        :return: the scalar value
        """
        return self._nn.predict(np.concatenate((state,self.bin_actions[action])))

    def fit(self, s, a, y):
        """
        Fits the NN model
        :param s: NxS matrix of states
        :param a: N-dimensional array of action indexes
        :param y: N-dimensional array of targets
        """
        a = np.concatenate([self.bin_actions[int(ai)].reshape(1,4) for ai in a], axis=0)
        sa = np.concatenate((s,a), axis=1)
        self._fitted = True
        self._nn.fit(sa, y)

    def get_weights(self):
        """
        :return: A flat vector with all NN weights
        """
        weights = self._nn.get_weights()
        weights = [w.flatten() for w in weights]
        return np.concatenate(weights)

    def update_weights(self, weights):
        """
        Sets the NN weights given a flat vector
        :param weights: A vector with the NN weights
        """
        old_weights = self._nn.get_weights()
        sizes = np.array([w.size for w in old_weights])
        new_weights = np.split(weights, np.cumsum(sizes)[:-1])
        shapes = [w.shape for w in old_weights]
        for i in range(len(new_weights)):
            new_weights[i] = np.reshape(new_weights[i],shapes[i])
        self._nn.set_weights(new_weights)

