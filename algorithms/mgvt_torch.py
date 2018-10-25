import numpy as np
from misc.policies import EpsilonGreedy
from approximators.mlp_torch import MLPQFunction
from misc.buffer import Buffer
from misc import utils
import time

import torch

prior_eigen = None
cholesky_mask = None
prior_normal = None
posterior_normal = None
sigma_inv = None

def unpack(params, C, K):
    """Unpacks a parameter vector into c, mu, L"""
    c = params[:C]
    mus = params[C:C + C * K]
    mu = mus.reshape(C, K)
    Ls = params[C + C * K:]
    L = Ls.reshape(C, K, K)
    return c, mu, L


def pack(c, mu, L):
    """Packs c and mu and L into a parameter vector"""
    params = np.concatenate((c, mu.flatten(), L.flatten()), axis=0)
    return params


def clip(params, cholesky_clip, C, K):
    """Makes sure the Cholesky factor L is well-defined"""
    global cholesky_mask
    c, mu, L = unpack(params, C, K)
    cholesky_mask = np.eye(K, dtype=bool) if cholesky_mask is None else cholesky_mask
    for i in range(C):
        mask = np.logical_and(L[i, :, :] < cholesky_clip, cholesky_mask)
        L[i, :, :][mask] = cholesky_clip
        L[i, :, :][np.triu_indices(K, 1)] = 0
    return pack(c, mu, L)


def sample_posterior(params, C, K):
    """Samples a Q function from the posterior distribution"""
    c, mu, L = unpack(params, C, K)
    cluster = np.random.choice(np.arange(C), p=c)
    return np.dot(L[cluster, : , :], np.random.randn(K,)) + mu[cluster, :]


def normal_KL(c, mu, Sigma, c_bar, mu_bar, Sigma_bar, L=None, precision=None):
    """ Computes the KL between normals for two GMMs """
    global prior_eigen
    Sigma_bar_inv = np.linalg.inv(Sigma_bar) if precision is None else precision
    inv_b = Sigma_bar_inv[np.newaxis]
    if prior_eigen is None:
        prior_eigen, _ = np.linalg.eig(Sigma_bar[np.newaxis])

    if L is None:
        posterior_eigen, _ = np.linalg.eig(Sigma[:, np.newaxis])
    else:
        posterior_eigen = np.diagonal(L[:, np.newaxis], axis1=2, axis2=3)**2
    posterior_eigen = np.real(posterior_eigen)
    posterior_eigen[posterior_eigen < 0] = 1e-10
    mu_diff = mu[:, np.newaxis] - mu_bar[np.newaxis]

    return 0.5 * (np.sum(np.log(prior_eigen / posterior_eigen) + posterior_eigen / prior_eigen, axis=2) +
                  np.matmul(np.matmul(mu_diff[:, :, np.newaxis], inv_b), mu_diff[:, :, :, np.newaxis])[:, :, 0, 0] -
                  mu.shape[1])


def normal_KL3(c, mu, Sigma, c_bar, mu_bar, Sigma_bar, L=None, precision=None):
    """ Computes the KL between normals for two GMMs using pytorch tensors """
    #mu, sigma, L must be torch tensors
    global prior_eigen
    Sigma_bar_inv = np.linalg.inv(Sigma_bar) if precision is None else precision
    inv_b = torch.from_numpy(Sigma_bar_inv[np.newaxis])
    if prior_eigen is None:
        prior_eigen_torch = torch.from_numpy(np.real(np.linalg.eig(Sigma_bar[np.newaxis])[0]))
    else:
        prior_eigen_torch = torch.from_numpy(prior_eigen)

    diag_mask = torch.eye(mu.shape[1]).unsqueeze(0).unsqueeze(0).double()
    posterior_eigen = ((L.unsqueeze(1)*diag_mask).sum(dim=-1))**2
    posterior_eigen[posterior_eigen < 0] = 1e-10
    mu_diff = mu.unsqueeze(1) - torch.from_numpy(mu_bar[np.newaxis])
    return 0.5 * (((prior_eigen_torch / posterior_eigen).log() + posterior_eigen / prior_eigen_torch).sum(dim=2) +
                  torch.matmul(torch.matmul(mu_diff.unsqueeze(2), inv_b), mu_diff.unsqueeze(3))[:, :, 0, 0] -
                  mu.shape[1])


def tight_ukl(c, mu, Sigma, c_bar, mu_bar, Sigma_bar, phi, psi, L=None,eps=0.01, max_iter=100, normalkl=None):
    """ Solves Variational problem to tight the upper bound"""

    kl = normal_KL(c, mu, Sigma, c_bar, mu_bar, Sigma_bar, L=L) if normalkl is None else normalkl
    # Constants
    e_kl = np.exp(-kl) + 1e-10
    i = 0
    done = False
    while i < max_iter and not done:

        psi_bar = c_bar[np.newaxis] * phi / np.sum(phi, axis=0, keepdims=True)
        done = np.max(np.abs(psi - psi_bar)) < eps
        psi = psi_bar

        phi_bar = c[:, np.newaxis] * psi * e_kl / np.sum(psi * e_kl, axis=1, keepdims=True)
        done = done and np.max(np.abs(phi-phi_bar)) < eps
        phi = phi_bar

        i += 1

    return phi, psi


def UKL(c, mu, Sigma, c_bar, mu_bar, Sigma_bar, phi, psi, precision=None, L=None):
    """Computes an upper bound to the KL"""
    kl = normal_KL(c, mu, Sigma, c_bar, mu_bar, Sigma_bar, L=L, precision=precision)   # (posterior, prior)

    kl_var = np.sum(phi * np.log(phi/psi))
    kl_gaussian = np.sum(phi * kl)
    return kl_var + kl_gaussian

def UKL3(c, mu, Sigma, c_bar, mu_bar, Sigma_bar, phi, psi, precision=None, L=None, normalkl=None):
    """Computes an upper bound to the KL"""
    kl = normal_KL3(c, mu, Sigma, c_bar, mu_bar, Sigma_bar, L=L, precision=precision) if normalkl is None \
         else normalkl # (posterior, prior)
    kl_var = np.sum(phi * np.log(phi/psi))
    kl_gaussian = (torch.from_numpy(phi) * kl).sum()
    return kl_var.item() + kl_gaussian


def sample_gmm(n_samples, c, mu, L):
    """ Samples a mixture of Gaussians """
    vs = np.random.randn(n_samples, mu.shape[1])
    clusters = np.random.choice(np.arange(c.size), n_samples, p=c)
    ws = np.matmul(vs[:,np.newaxis], np.transpose(L[clusters], (0,2,1)))[:,0,:] + mu[clusters]
    return ws, vs


def objective(samples, params, Q, c_bar, mu_bar, Sigma_bar, operator, n_samples, phi, psi, n_weights, lambda_, C, K,
              precision=None):
    """Computes the negative ELBO"""
    c, mu, L = unpack(params, C, K)
    # We add a small constant to make sure Sigma is always positive definite
    L_reg = L+0.1*np.eye(K)[np.newaxis]
    weights, _ = sample_gmm(n_weights, c, mu, L)
    likelihood = operator.expected_bellman_error(Q, samples, weights)
    assert likelihood >= 0
    kl = UKL(c, mu, None, c_bar, mu_bar, Sigma_bar, phi, psi, L=L_reg, precision=precision)
    assert kl >= 0
    return likelihood + lambda_ * kl / n_samples


def gradient_KL(c, mu, L, c_bar, mu_bar, Sigma_bar, phi, psi, max_iter_ukl, C, K, precision=None, tight_bound=True):
    # Cache the pairwise KL
    mu = torch.from_numpy(mu).requires_grad_()
    L = torch.from_numpy(L).requires_grad_()
    normalkl = normal_KL3(c, mu, None, c_bar, mu_bar, Sigma_bar, L=L, precision=precision)

    if tight_bound:
        psi = c[:, np.newaxis] * c_bar[np.newaxis]
        phi = np.array(psi)
        phi, psi = tight_ukl(c, mu, None, c_bar, mu_bar, Sigma_bar, phi, psi,
                             L=L,
                             max_iter=max_iter_ukl,
                             normalkl=normalkl.detach().numpy())

    assert np.all(np.logical_and(phi >= 0, psi >= 0))

    grad_c = np.zeros(C)
    ukl = UKL3(c, mu, None, c_bar, mu_bar, Sigma_bar, phi, psi, precision=precision, L=L, normalkl=normalkl)
    ukl.backward()
    grad_mu = mu.grad.data.numpy()
    grad_L = L.grad.data.numpy()
    return grad_c, grad_mu, grad_L, phi, psi


def init_posterior(c, mu, L, c_bar, mu_bar, Sigma_bar, phi, psi, C, K, cholesky_clip, max_iter_ukl, precision=None,
                   max_iter=10000, eta=1e-5, eps=0.00001, ukl_tight_freq=100):
    i = 0
    ukl_prev = UKL(c, mu, None, c_bar, mu_bar, Sigma_bar, phi, psi, L=L)
    done = False
    params = pack(c, mu, L)
    while not done and i < max_iter:
        if i % ukl_tight_freq == 0:
            grad_c, grad_mu, grad_L, phi, psi = gradient_KL(c, mu, L, c_bar, mu_bar, Sigma_bar, phi, psi,
                                                            max_iter_ukl, C, K, precision=precision)
        else:
            grad_c, grad_mu, grad_L, phi, psi = gradient_KL(c, mu, L, c_bar, mu_bar, Sigma_bar, phi, psi, max_iter_ukl,
                                                            C, K, precision=precision, tight_bound=False)
        params = clip(params - eta * pack(grad_c, grad_mu, grad_L), cholesky_clip, C, K)
        c, mu, L = unpack(params, C, K)
        ukl = UKL(c, mu, None, c_bar, mu_bar, Sigma_bar, phi, psi, L=L, precision=precision)
        done = np.abs(ukl-ukl_prev) < eps
        ukl_prev = ukl
        i += 1
        print("Initializing prior %d... UKL: %f" % (i, ukl))
    return params, phi, psi


def gradient(samples, params, Q, c_bar, mu_bar, Sigma_bar, operator, n_samples, phi, psi, n_weights, lambda_,
             max_iter_ukl, C, K, precision=None, t_step=0, ukl_tight_freq=1):
    """Computes the objective function gradient"""
    c, mu, L = unpack(params, C, K)
    grad_c = np.zeros(c.shape)

    _, vs = utils.sample_mvn(n_weights * C, mu[0, :], L[0, :, :])

    ws = np.matmul(vs.reshape(C,n_weights,K), np.transpose(L, (0,2,1))) + mu[:, np.newaxis]
    be_grad = operator.gradient_be(Q, samples, ws.reshape(C*n_weights, K)).reshape(C, n_weights, K)
    # Gradient of the expected Bellman error wrt mu
    ebe_grad_mu = np.average(be_grad, axis=1)
    # Gradient of the expected Bellman error wrt L.
    ebe_grad_L = np.average(np.matmul(be_grad[:, :, :, np.newaxis], vs.reshape(C, n_weights, K)[:,:,np.newaxis]), axis=1)
    ebe_grad_mu = c[:, np.newaxis] * ebe_grad_mu
    ebe_grad_L = c[:, np.newaxis, np.newaxis] * ebe_grad_L

    kl_grad_c, kl_grad_mu, kl_grad_L, phi, psi = gradient_KL(c, mu, L, c_bar, mu_bar, Sigma_bar, phi, psi,
                                                             max_iter_ukl, C, K, precision=precision,
                                                             tight_bound=(t_step % ukl_tight_freq == 0))
    grad_mu = ebe_grad_mu + lambda_ * kl_grad_mu / n_samples
    grad_L = ebe_grad_L + lambda_ * kl_grad_L / n_samples

    return pack(grad_c, grad_mu, grad_L)


def learn(mdp,
          Q,
          operator,
          max_iter=5000,
          buffer_size=10000,
          batch_size=50,
          alpha_adam=0.001,
          alpha_sgd=0.1,
          lambda_=0.001,
          n_weights=10,
          train_freq=1,
          eval_freq=50,
          random_episodes=0,
          eval_states=None,
          eval_episodes=1,
          mean_episodes=50,
          preprocess=lambda x: x,
          cholesky_clip=0.0001,
          bandwidth=0.00001,
          post_components=1,
          max_iter_ukl=60,
          eps=0.001,
          eta=1e-6,
          time_coherent=False,
          n_source=10,
          source_file=None,
          seed=None,
          render=False,
          verbose=True,
          ukl_tight_freq=1,
          sources=None):

    if seed is not None:
        np.random.seed(seed)

    # Randomly initialize the weights in case an MLP is used
    if isinstance(Q, MLPQFunction):
        Q.init_weights()

    # Reset global variables
    global prior_eigen
    prior_eigen = None
    global cholesky_mask
    cholesky_mask = None
    global prior_normal
    prior_normal = None
    global posterior_normal
    posterior_normal = None

    # Initialize policies
    pi_g = EpsilonGreedy(Q, np.arange(mdp.action_space.n), epsilon=0)

    # Get number of features
    K = Q._w.size
    C = post_components

    # Load weights and construct prior distribution
    weights = utils.load_object(source_file) if sources is None else sources
    ws = np.array([w[1] for w in weights])
    np.random.shuffle(ws)
    # Take only the first n_source weights
    ws = ws[:n_source, :]
    mu_bar = ws
    Sigma_bar = np.tile(np.eye(K) * bandwidth, (n_source,1,1))
    Sigma_bar_inv = np.tile((1/bandwidth * np.eye(K))[np.newaxis], (n_source, 1, 1))
    c_bar = np.ones(n_source)/n_source

    # We initialize the parameters of the posterior to the best approximation of the posterior family to the prior
    c = np.ones(C) / C
    psi = c[:, np.newaxis] * c_bar[np.newaxis]
    phi = np.array(psi)

    mu = np.array([100 * np.random.randn(K) for _ in range(C)])
    Sigma = np.array([np.eye(K) for _ in range(C)])

    phi, psi = tight_ukl(c, mu, Sigma, c_bar, mu_bar, Sigma_bar, phi, psi, max_iter=max_iter_ukl, eps=eps)
    params, phi, psi = init_posterior(c, mu, Sigma, c_bar, mu_bar, Sigma_bar, phi, psi, C, K, cholesky_clip,
                                      max_iter_ukl, max_iter=max_iter_ukl * 10, precision=Sigma_bar_inv, eta=eta, eps=eps)

    # Add random episodes if needed
    init_samples = list()
    if random_episodes > 0:
        w, _ = sample_gmm(random_episodes, c_bar, mu_bar, np.sqrt(Sigma_bar))
        for i in range(random_episodes):
            Q._w = w[i]
            init_samples.append(utils.generate_episodes(mdp, pi_g, n_episodes=1, preprocess=preprocess))
        init_samples = np.concatenate(init_samples)

        t, s, a, r, s_prime, absorbing, sa = utils.split_data(init_samples, mdp.state_dim, mdp.action_dim)
        init_samples = np.concatenate((t[:, np.newaxis], preprocess(s), a, r[:, np.newaxis], preprocess(s_prime),
                                       absorbing[:, np.newaxis]), axis=1)

    # Figure out the effective state-dimension after preprocessing is applied
    eff_state_dim = preprocess(np.zeros(mdp.state_dim)).size

    # Create replay buffer
    buffer = Buffer(buffer_size, eff_state_dim)
    n_init_samples = buffer.add_all(init_samples) if random_episodes > 0 else 0

    # Results
    iterations = []
    episodes = []
    n_samples = []
    evaluation_rewards = []
    learning_rewards = []
    episode_rewards = [0.0]
    l_2 = []
    l_inf = []
    fvals = []
    episode_t = []

    # Create masks for ADAM and SGD
    adam_mask = pack(np.zeros(C), np.ones((C,K)) * alpha_adam, np.zeros((C,K,K)))  # ADAM learns only \mu
    sgd_mask = pack(np.zeros(C), np.zeros((C,K)), np.ones((C,K,K)) * alpha_sgd)  # SGD learns only L

    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0

    # Init env
    s = mdp.reset()
    h = 0
    Q._w = sample_posterior(params, C, K)

    start_time = time.time()

    # Learning
    for i in range(max_iter):

        # If we do not use time coherent exploration, resample parameters
        Q._w = sample_posterior(params, C, K) if not time_coherent else Q._w
        # Take greedy action wrt current Q-function
        s_prep = preprocess(s)
        a = np.argmax(Q.value_actions(s_prep))
        # Step
        s_prime, r, done, _ = mdp.step(a)
        # Build the new sample and add it to the dataset
        buffer.add_sample(h, s_prep, a, r, preprocess(s_prime), done)

        # Take a step of gradient if needed
        if i % train_freq == 0:
            # Estimate gradient
            g = gradient(buffer.sample_batch(batch_size), params, Q, c_bar, mu_bar, Sigma_bar, operator,
                         i + 1, phi, psi, n_weights, lambda_, max_iter_ukl, C, K, precision=Sigma_bar_inv,
                         t_step=i, ukl_tight_freq=ukl_tight_freq)

            # Take a gradient step for \mu
            params, t, m_t, v_t = utils.adam(params, g, t, m_t, v_t, alpha=adam_mask)
            # Take a gradient step for L
            params = utils.sgd(params, g, alpha=sgd_mask)
            # Clip parameters
            params = clip(params, cholesky_clip, C, K)

        # Add reward to last episode
        episode_rewards[-1] += r * mdp.gamma ** h

        s = s_prime
        h += 1
        if done or h >= mdp.horizon:

            episode_rewards.append(0.0)
            s = mdp.reset()
            h = 0
            Q._w = sample_posterior(params, C, K)
            episode_t.append(i)

        # Evaluate model
        if i % eval_freq == 0:

            #Save current weights
            current_w = np.array(Q._w)

            # Evaluate MAP Q-function
            c, mu, _ = unpack(params, C, K)
            rew = 0
            for j in range(C):
                Q._w = mu[j]
                rew += utils.evaluate_policy(mdp, pi_g, render=render, initial_states=eval_states,
                                             n_episodes=eval_episodes, preprocess=preprocess)[0]
            rew /= C

            learning_rew = np.mean(episode_rewards[-mean_episodes - 1:-1]) if len(episode_rewards) > 1 else 0.0
            br = operator.bellman_residual(Q, buffer.sample_batch(batch_size)) ** 2
            l_2_err = np.average(br)
            l_inf_err = np.max(br)
            fval = objective(buffer.sample_batch(batch_size), params, Q, c_bar, mu_bar, Sigma_bar, operator,
                             i + 1, phi, psi, n_weights, lambda_, C, K, precision=Sigma_bar_inv)

            # Append results
            iterations.append(i)
            episodes.append(len(episode_rewards) - 1)
            n_samples.append(n_init_samples + i + 1)
            evaluation_rewards.append(rew)
            learning_rewards.append(learning_rew)
            l_2.append(l_2_err)
            l_inf.append(l_inf_err)
            fvals.append(fval)

            # Make sure we restart from s
            mdp.reset(s)

            # Restore weights
            Q._w = current_w

            end_time = time.time()
            elapsed_time = end_time - start_time
            start_time = end_time

            if verbose:
                print("Iter {} Episodes {} Rew(G) {} Rew(L) {} Fval {} L2 {} L_inf {} time {:.1f} s".format(
                    i, episodes[-1], rew, learning_rew, fval, l_2_err, l_inf_err, elapsed_time))

    run_info = [iterations, episodes, n_samples, learning_rewards, evaluation_rewards, l_2, l_inf, fvals, episode_rewards[:len(episode_t)], episode_t]
    weights = np.array(mu)

    return [mdp.get_info(), weights, run_info]
