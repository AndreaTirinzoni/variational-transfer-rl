import sys
sys.path.append("../")

import numpy as np
from policies import EpsilonGreedy
from approximators.mlp_torch import MLPQFunction
from buffer import Buffer
import utils
import time


def unpack(params, K):
    """Unpacks a parameter vector into mu and L"""
    mu = params[:K]
    L = params[K:].reshape(K,K)
    return mu, L


def pack(mu, L):
    """Packs mu and L into a parameter vector"""
    return np.concatenate((mu, L.flatten()))


def clip(params, cholesky_clip, K):
    """Makes sure the Cholensky factor L is well-defined"""
    mu, L = unpack(params, K)
    # Clip the diagonal to 0.01
    mask = np.logical_and(L < cholesky_clip, np.eye(K, dtype=bool))
    L[mask] = cholesky_clip
    # Make sure L is lower triangular
    L[np.triu_indices(K, 1)] = 0
    return pack(mu, L)


def sample_posterior(params, K):
    """Samples a Q function from the posterior distribution"""
    mu, L = unpack(params, K)
    return np.dot(L, np.random.randn(K,)) + mu


def objective(samples, params, Q, mu_bar, Sigma_bar_inv, operator, n_samples, lambda_, n_weights):
    """Computes the negative ELBO"""
    mu, L = unpack(params, Q._w.size)
    # We add a small constant to make sure Sigma is always positive definite
    Sigma = np.dot(L, L.T)
    weights, _ = utils.sample_mvn(n_weights, mu, L)
    likelihood = operator.expected_bellman_error(Q, samples, weights)
    assert likelihood >= 0
    kl = utils.KL(mu, Sigma, mu_bar, Sigma_bar_inv)
    assert kl >= 0
    return likelihood + lambda_ * kl / n_samples


def gradient(samples, params, Q, mu_bar, Sigma_bar_inv, operator, n_samples, lambda_, n_weights):
    """Computes the objective function gradient"""
    mu, L = unpack(params, Q._w.size)
    ws, vs = utils.sample_mvn(n_weights, mu, L)
    be_grad = operator.gradient_be(Q, samples, ws)
    # Gradient of the expected Bellman error wrt mu
    ebe_grad_mu = np.average(be_grad, axis=0)
    # Gradient of the expected Bellman error wrt L.
    ebe_grad_L = np.average(be_grad[:, :, np.newaxis] * vs[:, np.newaxis, :], axis=0)
    kl_grad_mu, kl_grad_L = utils.gradient_KL(mu, L, mu_bar, Sigma_bar_inv)
    grad_mu = ebe_grad_mu + lambda_ * kl_grad_mu / n_samples
    grad_L = ebe_grad_L + lambda_ * kl_grad_L / n_samples

    return pack(grad_mu, grad_L)


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
          sigma_reg=0.0001,
          cholesky_clip=0.0001,
          time_coherent=False,
          n_source=10,
          source_file=None,
          seed=None,
          render=False,
          verbose=True,
          sources=None):

    if seed is not None:
        np.random.seed(seed)

    # Randomly initialize the weights in case an MLP is used
    if isinstance(Q, MLPQFunction):
        Q.init_weights()

    # Initialize policies
    pi_g = EpsilonGreedy(Q, np.arange(mdp.action_space.n), epsilon=0)

    # Get number of features
    K = Q._w.size

    # Load weights and construct prior distribution
    weights = utils.load_object(source_file) if sources is None else sources
    ws = np.array([w[1] for w in weights])
    np.random.shuffle(ws)
    # Take only the first n_source weights
    ws = ws[:n_source, :]
    mu_bar = np.mean(ws, axis=0)
    Sigma_bar = np.cov(ws.T)
    # We use higher regularization for the prior to prevent the ELBO from diverging
    Sigma_bar_inv = np.linalg.inv(Sigma_bar + np.eye(K) * sigma_reg)
    # We initialize the parameters at the prior with smaller regularization (just to make sure Sigma_bar is pd)
    params = clip(pack(mu_bar, np.linalg.cholesky(Sigma_bar + np.eye(K) * cholesky_clip**2)), cholesky_clip, K)

    # Add random episodes if needed
    if random_episodes > 0:
        init_samples = list()
        for i in range(random_episodes):
            Q._w = sample_posterior(params, K)
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

    # Create masks for ADAM and SGD
    adam_mask = pack(np.ones(K) * alpha_adam, np.zeros((K,K)))  # ADAM learns only \mu
    sgd_mask = pack(np.zeros(K), np.ones((K,K)) * alpha_sgd)  # SGD learns only L

    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0

    # RMSprop for Variance
    v_t_var = 0.

    # Init env
    s = mdp.reset()
    h = 0
    Q._w = sample_posterior(params, K)

    start_time = time.time()

    # Learning
    for i in range(max_iter):

        # If we do not use time coherent exploration, resample parameters
        Q._w = sample_posterior(params, K) if not time_coherent else Q._w
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
            g = gradient(buffer.sample_batch(batch_size), params, Q, mu_bar, Sigma_bar_inv, operator,
                         i + 1, lambda_, n_weights)
            # Take a gradient step for \mu
            params, t, m_t, v_t = utils.adam(params, g, t, m_t, v_t, alpha=adam_mask)
            # Take a gradient step for L
            params = utils.sgd(params, g, alpha=sgd_mask)
            # params,v_t_var = utils.rmsprop(params, g, v_t_var, alpha=sgd_mask)
            # Clip parameters
            params = clip(params, cholesky_clip, K)

        # Add reward to last episode
        episode_rewards[-1] += r * mdp.gamma ** h

        s = s_prime
        h += 1
        if done or h >= mdp.horizon:

            episode_rewards.append(0.0)
            s = mdp.reset()
            h = 0
            Q._w = sample_posterior(params, K)

        # Evaluate model
        if i % eval_freq == 0:

            #Save current weights
            current_w = np.array(Q._w)

            # Evaluate MAP Q-function
            mu, _ = unpack(params, K)
            Q._w = mu
            rew = utils.evaluate_policy(mdp, pi_g, render=render, initial_states=eval_states,
                                        n_episodes=eval_episodes, preprocess=preprocess)[0]
            learning_rew = np.mean(episode_rewards[-mean_episodes - 1:-1]) if len(episode_rewards) > 1 else 0.0
            br = operator.bellman_residual(Q, buffer.sample_batch(batch_size)) ** 2
            l_2_err = np.average(br)
            l_inf_err = np.max(br)
            fval = objective(buffer.sample_batch(batch_size), params, Q, mu_bar, Sigma_bar_inv, operator,
                             i + 1, lambda_, n_weights)

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

    run_info = [iterations, episodes, n_samples, learning_rewards, evaluation_rewards, l_2, l_inf, fvals]
    weights = np.array(mu)

    return [mdp.get_info(), weights, run_info]
