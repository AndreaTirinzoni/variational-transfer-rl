import numpy as np
import utils
import algorithms.e_greedy_policy as egreedy
import VariationalTransfer.Distributions as dist


"""
Variational Transfer using Gaussian Distributions for the Prior and Posterior distributions (with diagonal covariances)
"""

class VarTransferGaussian:

    def __init__(self, mdp, bellman_operator, prior, learning_rate=1, likelihood_weight=1e-3):
        self._bellman = bellman_operator
        self._mdp = mdp
        self._prior = prior
        self._l_rate = learning_rate
        self._likelihood = likelihood_weight
        self._posterior = dist.AnisotropicNormalPosterior()
        self._posterior.set_params(prior.get_params())

    def solve_task(self, max_iter=100, n_fit=1, batch_size=20, nsamples_for_estimation=100, render=False, verbose=False):
        Q = self._bellman.get_Q()
        pol_g = egreedy.eGreedyPolicy(Q, Q.actions)
        samples = utils.generate_episodes(self._mdp, pol_g, n_episodes=1, render=False)
        performance = list()
        elbo = list()
        for i in range(max_iter):
            Q.update_weights(self._posterior.sample()[0, :])
            new_samples = utils.generate_episodes(self._mdp, pol_g, n_episodes=batch_size, render=False)
            samples = np.vstack((samples, new_samples))
            for _ in range(n_fit):
                elbo.append(self._compute_ELBO())
                grad = self._compute_evidence_gradient(samples[:, 1:], nsamples_for_estimation) + self._compute_KL_gradient(samples[:, 1:])
                self._posterior.grad_step(self._learning_rate() * grad)
            rew = utils.evaluate_policy(self._mdp, pol_g, render=render, initial_states=np.array([0, 0])) #TODO add parameter.
            performance.append(rew)

            if verbose:
                print("===============================================")
                print("Iteration " + str(i))
                print("Reward: " + str(rew))
                print("===============================================")

        return np.array(performance), np.array(elbo)

    def _compute_KL_gradient(self, samples):
        prior_params = self._prior.get_params()
        midpoint = int(prior_params.size/2)
        prior_mean = prior_params[0:midpoint]
        prior_covar = prior_params[midpoint:]

        posterior_params = self._posterior.get_params()
        midpoint = int(posterior_params.size / 2)
        posterior_mean = posterior_params[0:midpoint]
        posterior_covar = posterior_params[midpoint: ]

        grad_mean = 1/prior_covar * (posterior_mean - prior_mean)
        grad_covar = 0.5 * (1/prior_covar - 1/posterior_covar)
        return np.hstack((grad_mean, grad_covar))


    def _compute_ELBO(self):
        return 0    # TODO implement ELBO

    def _compute_evidence_gradient(self, data, nsamples=1):
        samples = self._posterior.sample(nsamples)
        grad, diag_hessian = self._bellman.compute_gradient_diag_hessian(data, samples)
        grad = self._likelihood * data.shape[0] * np.average(grad, axis=1)
        diag_hessian = 0.5 * self._likelihood * data.shape[0] * np.average(diag_hessian, axis=1)
        return np.hstack((grad, diag_hessian))

    def _learning_rate(self):
        return self._l_rate


