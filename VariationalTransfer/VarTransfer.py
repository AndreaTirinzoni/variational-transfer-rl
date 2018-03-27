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
        self._gradient = 0.
        self._gradient2 = 0.

    def solve_task(self, max_iter=1000, n_fit=1, batch_size=20, nsamples_for_estimation=100, render=False, verbose=False):
        Q = self._bellman.get_Q()
        pol_g = egreedy.eGreedyPolicy(Q, Q.actions)
        samples = utils.generate_episodes(self._mdp, pol_g, n_episodes=1, render=False)
        performance = list()
        elbo = list()
        brs = list()

        t = 0
        for i in range(max_iter):
            Q.update_weights(self._posterior.sample()[0, :])
            new_samples = utils.generate_episodes(self._mdp, pol_g, n_episodes=batch_size, render=False)
            samples = np.vstack((samples, new_samples))
            for k in range(n_fit):
                grad = self._compute_evidence_gradient(samples[:, 1:], nsamples_for_estimation) + self._compute_KL_gradient(samples[:, 1:])

                # step = self._adam(grad, t)
                # self._posterior.grad_step(step)
                # t = t+1

                self._posterior.grad_step(self._learning_rate() * grad)

            Q.update_weights(self._posterior.get_mean())
            rew, _, _, _ = utils.evaluate_policy(self._mdp, pol_g, render=render, initial_states=np.array([0., 0.])) #TODO add parameter.
            performance.append(rew)
            elbo.append(self._compute_ELBO(samples[:, 1:], nsamples_for_estimation))
            ws = self._posterior.sample(200)
            br = self._bellman.bellman_residual(samples[:, 1:], weights=ws)
            brs.append(br)


            if verbose:
                utils.plot_Q(Q, tuple(self._mdp.size))
                print("===============================================")
                print("Iteration " + str(i))
                print("Reward Mean Q: " + str(rew))
                ws = self._posterior.sample(200)
                br = self._bellman.bellman_residual(samples[:, 1:], weights=ws)
                print("Bellman Residual Avg: " + str(np.average(np.average(br**2, axis=1))))
                print("Bellman Residual: " + str(np.average(self._bellman.bellman_residual(samples[:, 1:])**2)))
                print("ELBO: " + str(elbo[len(elbo)-1]))
                print("Mean Var: " + str(np.average(self._posterior.get_variance())))
                print("Min Var: " + str(np.min(self._posterior.get_variance())))
                print("Max Var: " + str(np.max(self._posterior.get_variance())))
                print("===============================================")

        return np.array(performance), np.array(elbo), np.array(brs)

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


    def _compute_ELBO(self, data, nsamples):
        samples = self._posterior.sample(nsamples)
        br = self._bellman.bellman_residual(data, weights=samples)
        br = self._likelihood * np.average(np.sum(br**2, axis=0))

        prior_params = self._prior.get_params()
        midpoint = int(prior_params.size / 2)
        prior_mean = prior_params[0:midpoint]
        prior_covar = prior_params[midpoint:]

        posterior_params = self._posterior.get_params()
        K = int(posterior_params.size / 2)
        posterior_mean = posterior_params[0:K]
        posterior_covar = posterior_params[K:]

        kl = .5 * (np.log(np.prod(prior_covar)/np.prod(posterior_covar)) + np.sum(posterior_covar/prior_covar) \
                          + np.dot((prior_mean - posterior_mean) / prior_covar, (prior_mean - posterior_mean)) \
                   - K)

        return br + kl

    def _compute_evidence_gradient(self, data, nsamples=1):
        samples = self._posterior.sample(nsamples)
        grad, diag_hessian = self._bellman.compute_gradient_diag_hessian(data, samples)
        grad = self._likelihood * data.shape[0] * np.average(grad, axis=1)
        diag_hessian = 0.5 * self._likelihood * data.shape[0] * np.average(diag_hessian, axis=1)
        return np.hstack((grad, diag_hessian))

    def _learning_rate(self):
        lr = np.zeros(self._prior.get_params().size)
        # lr[0:int(lr.size/2)] += self._l_rate
        lr += self._l_rate
        return lr

    def _adam(self, gradient, step, beta1=0.9, beta2=0.99, learning_rate=1e-3, eps=1e-8):
        self._gradient = beta1 * self._gradient + (1-beta1) * gradient
        self._gradient2 = beta2 * self._gradient2 + (1-beta2) * gradient**2
        self._gradient /= (1-beta1**step)
        self._gradient2 /= (1-beta2**step)
        return learning_rate * self._gradient/np.sqrt(self._gradient2 + eps)

    def reset(self):
        self._posterior.set_params(self._prior.get_params())


