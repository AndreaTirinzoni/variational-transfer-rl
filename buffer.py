import numpy as np


class Buffer:
    """A replay buffer with efficient sample addition"""

    def __init__(self, capacity, state_dim):
        # The columns are (t,s,a,r,s_prime,done)
        self._buffer = np.zeros((capacity, 1 + state_dim + 1 + 1 + state_dim + 1))
        self._next = 0
        self._capacity = capacity
        self.n_samples = 0
        self._a_idx = 1 + state_dim
        self._r_idx = self._a_idx + 1
        self._s_idx = self._r_idx + 1

    def add_sample(self, t, s, a, r, s_prime, done):
        """Adds a single sample to the buffer. Returns the updated number of samples"""
        self._buffer[self._next, 0] = t
        self._buffer[self._next, 1:self._a_idx] = s
        self._buffer[self._next, self._a_idx] = a
        self._buffer[self._next, self._r_idx] = r
        self._buffer[self._next, self._s_idx:-1] = s_prime
        self._buffer[self._next, -1] = 1 if done else 0

        self._next = self._next + 1 if self._next < self._capacity - 1 else 0
        self.n_samples = min(self.n_samples + 1, self._capacity)
        return self.n_samples

    def add_all(self, dataset):
        """Adds all samples in dataset to the buffer. Returns the updated number of samples"""
        # TODO this function can be made more efficient. Anyway, it should be used only for initialization
        for i in range(dataset.shape[0]):
            self.add_sample(dataset[i, 0], dataset[i, 1:self._a_idx], dataset[i, self._a_idx], dataset[i, self._r_idx],
                            dataset[i, self._s_idx:-1], dataset[i, -1])
        return self.n_samples

    def sample_batch(self, batch_size):
        """Samples a minibatch from the replay buffer"""
        idx = np.random.randint(self.n_samples, size=batch_size)
        return self._buffer[idx, :]