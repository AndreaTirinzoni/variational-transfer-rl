import numpy as np
import utils
import algorithms.e_greedy_policy as egreedy

class VarTransferGaussian:

    def __init__(self, mdp, bellman_operator, prior, learning_rate=1, likelihood_weight=1e-3):
        self._bellman = bellman_operator
        self._mdp = mdp
        self._prior = prior
        self._l_rate = learning_rate
        self._likelihood = likelihood_weight


    def solve_task(self, max_iter=100, n_fit=1, batch_size=20, render=False):
        Q = self._bellman.get_Q()
        pol_g = egreedy.eGreedyPolicy(Q, Q._actions)
        samples = np.empty((1, 2*Q.state_dim + Q.action_dim + 2))
        #Initialize Posterior to Prior. self._posterior
        performance = list()
        elbo = list()
        for _ in range(max_iter):
            Q.update_weights(self._posterior.sample())
            new_samples = utils.generate_episodes(self._mdp, pol_g, n_episodes=batch_size, render=False)
            samples = np.vstack((samples, new_samples))
            for _ in range(n_fit):
                elbo.append(self._compute_ELBO())
                performance.append(utils.evaluate_policy(self._mdp, pol_g, render=render))
                grad = self._compute_evidence_gradient(samples) + self._compute_KL_gradient(samples)
                self._posterior.grad_step(self._learning_rate() * grad)

        return np.array(performance), np.array(elbo)

    def _compute_KL_gradient(self, samples):
        return np.zeros(100)

    def _compute_ELBO(self):
        return 0

    def _compute_evidence_gradient(self, samples):
        return np.zeros(100)

    def _learning_rate(self):
        return self._l_rate