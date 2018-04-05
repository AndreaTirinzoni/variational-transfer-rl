from abc import abstractmethod, ABCMeta

import numpy as np
import utils
import algorithms.e_greedy_policy as egreedy
import VariationalTransfer.Distributions as dist
import VariationalTransfer.sampled_policy as sp

class VarTransfer(metaclass=ABCMeta):

    def __init__(self, mdp, bellman_operator, prior, learning_rate=1., likelihood_weight=1e-3):
        self._bellman = bellman_operator
        self._mdp = mdp
        self._prior = prior
        self._l_rate = learning_rate
        self._likelihood = likelihood_weight
        self._gradient = 0.
        self._gradient2 = 0.

    def _step(self, gradient):
        return self._l_rate * gradient

    def _adam(self, gradient, step, beta1=0.9, beta2=0.99, learning_rate=1e-3, eps=1e-8):
        self._gradient = beta1 * self._gradient + (1-beta1) * gradient
        self._gradient2 = beta2 * self._gradient2 + (1-beta2) * gradient**2
        grad_hat = self._gradient/(1.-(beta1**step))
        grad2_hat = self._gradient2/(1.-(beta2**step))
        return learning_rate * grad_hat/np.sqrt(grad2_hat + eps)

    def reset(self):
        self._posterior.set_params(self._prior.get_params())

    @abstractmethod
    def _compute_KL_gradient(self, samples):
        pass

    @abstractmethod
    def _compute_ELBO(self, data, nsamples):
        pass

    @abstractmethod
    def _compute_evidence_gradient(self, data, nsamples):
        pass

    @abstractmethod
    def _generate_episode(self, batch_size, render=False):
        pass

    def solve_task(self, max_iter=1000, n_fit=1, batch_size=20, nsamples_for_estimation=100, render=False, verbose=False, adam=False):
        samples = self._generate_episode(batch_size, render=False)
        performance = list()
        elbo = list()
        brs = list()
        Q = self._bellman.get_Q()
        pol_g = egreedy.eGreedyPolicy(Q, Q.actions)

        t = 0
        if adam:
            self._gradient2 = 0.
            self._gradient = 0.
            
        for i in range(max_iter):
            new_samples = self._generate_episode(batch_size)
            samples = np.vstack((samples, new_samples))
            for k in range(n_fit):
                grad = self._compute_evidence_gradient(samples[:, 1:], nsamples_for_estimation) + self._compute_KL_gradient(samples[:, 1:])
                if not adam:
                    self._posterior.grad_step(self._step(grad))
                else:
                    self._posterior.grad_step(self._adam(grad, t))

                t += 1

            Q.update_weights(self._posterior.get_mean())
            rew, _, _, _ = utils.evaluate_policy(self._mdp, pol_g, render=render, initial_states=np.array([0., 0.])) #TODO add parameter.
            performance.append(rew)
            elbo.append(self._compute_ELBO(samples[:, 1:], nsamples_for_estimation))
            ws = self._posterior.sample(nsamples_for_estimation)
            br = self._bellman.bellman_residual(samples[:, 1:], weights=ws)
            brs.append(np.average(np.average(br**2, axis=1)))

            if verbose:
                utils.plot_Q(Q, tuple(self._mdp.size))
                print("===============================================")
                print("Iteration " + str(i))
                print("Reward Mean Q: " + str(rew))
                ws = self._posterior.sample(nsamples_for_estimation)
                br = self._bellman.bellman_residual(samples[:, 1:], weights=ws)
                print("Bellman Residual Avg: " + str(np.average(np.average(br**2, axis=1))))
                print("Bellman Residual: " + str(np.average(self._bellman.bellman_residual(samples[:, 1:])**2)))
                print("ELBO: " + str(elbo[len(elbo)-1]))
                print("Mean Var: " + str(np.average(self._posterior.get_variance())))
                print("Min Var: " + str(np.min(self._posterior.get_variance())))
                print("Max Var: " + str(np.max(self._posterior.get_variance())))
                print("===============================================")

        return np.array(performance), np.array(elbo), np.array(brs)


"""
Variational Transfer using Gaussian Distributions for the Prior and Posterior distributions (with diagonal covariances)
"""
class VarTransferGaussian(VarTransfer):

    def __init__(self, mdp, bellman_operator, prior, learning_rate=1., likelihood_weight=1e-3, expected_pol=False):
        super(VarTransferGaussian, self).__init__(mdp, bellman_operator, prior, learning_rate, likelihood_weight)
        self._posterior = dist.AnisotropicNormalPosterior()
        self._posterior.set_params(prior.get_params())
        Q = self._bellman.get_Q()
        if not expected_pol:
            self._policy = egreedy.eGreedyPolicy(Q, Q.actions)
        else:
            self._policy = sp.expectedPolicy(Q, Q.actions, self._posterior)

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
        return np.hstack((grad_mean, grad_covar))/samples.shape[0]


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

        return (br + kl)/data.shape[0]

    def _compute_evidence_gradient(self, data, nsamples=1):
        samples = self._posterior.sample(nsamples)
        grad, diag_hessian = self._bellman.compute_gradient_diag_hessian(data, samples)
        grad = self._likelihood * np.average(grad, axis=1)
        diag_hessian = 0.5 * self._likelihood * np.average(diag_hessian, axis=1)
        return np.hstack((grad, diag_hessian))

    def _generate_episode(self, batch_size, render=False):
        self._bellman.get_Q().update_weights(self._posterior.sample(1)[0])
        return utils.generate_episodes(self._mdp, self._policy, n_episodes=batch_size, render=render)


"""
Variational Transferring with Multivariate Normal for Prior/Posterior distributions
"""
class VarTransferFullGaussian(VarTransfer):

    def __init__(self, mdp, bellman_operator, prior, learning_rate=1., likelihood_weight=1e-3, expected_pol=False):
        super(VarTransferFullGaussian, self).__init__(mdp, bellman_operator, prior, learning_rate, likelihood_weight)
        Q = self._bellman.get_Q()
        self._posterior = dist.NormalPosterior(Q.get_dim())
        self._posterior.set_params(prior.get_params())
        if not expected_pol:
            self._policy = egreedy.eGreedyPolicy(Q, Q.actions)
        else:
            self._policy = sp.expectedPolicy(Q, Q.actions, self._posterior)
        self._prior_prec = np.linalg.inv(prior.get_covar())

    def _generate_episode(self, batch_size, render=False):
        self._bellman.get_Q().update_weights(self._posterior.sample(1)[0])
        return utils.generate_episodes(self._mdp, self._policy, n_episodes=batch_size, render=render)

    def _compute_KL_gradient(self, samples):

        prior_mean = self._prior.get_mean()
        posterior_mean = self._posterior.get_mean()
        posterior_covar = self._posterior.get_covar()
        posterior_prec = np.linalg.inv(posterior_covar)
        grad_mean = np.dot(self._prior_prec,(posterior_mean - prior_mean))
        grad_covar = 0.5 * (self._prior_prec - posterior_prec)
        return np.hstack((grad_mean, np.ravel(grad_covar)))/samples.shape[0]

    def _compute_evidence_gradient(self, data, nsamples=1):
        samples = self._posterior.sample(nsamples)
        grad, hessian = self._bellman.compute_gradient_hessian(data, samples)
        grad = self._likelihood * np.average(grad, axis=1)
        hessian = 0.5 * self._likelihood * np.average(hessian, axis=2)
        return np.hstack((grad, np.ravel(hessian)))

    def _compute_ELBO(self, data, nsamples):
        samples = self._posterior.sample(nsamples)
        br = self._bellman.bellman_residual(data, weights=samples)
        br = self._likelihood * np.average(np.sum(br**2, axis=0))

        prior_covar = self._prior.get_covar()
        prior_mean = self._prior.get_mean()
        posterior_mean = self._posterior.get_mean()
        posterior_covar = self._posterior.get_covar()
        K = prior_mean.size

        x = np.linalg.det(prior_covar)
        y = np.linalg.det(posterior_covar)
        kl = .5 * (np.log(x/y) + np.trace(self._prior_prec @ posterior_covar) \
                          + np.dot((prior_mean - posterior_mean)[np.newaxis] @ self._prior_prec , (prior_mean - posterior_mean)) \
                   - K)

        return (br + kl)/data.shape[0]

