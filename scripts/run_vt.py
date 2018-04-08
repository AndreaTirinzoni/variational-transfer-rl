import sys
sys.path.append("../")

import numpy as np
from features.agrbf import AGaussianRBF
from VariationalTransfer.LinearQRegressor import LinearQRegressor
from envs.walled_gridworld import WalledGridworld
from algorithms.e_greedy_policy import eGreedyPolicy
import utils
import argparse
from joblib import Parallel, delayed
import datetime


def unpack(params):
    """Unpacks a parameter vector into mu and L"""
    mu = params[:K]
    L = params[K:].reshape(K,K)
    return mu, L


def pack(mu, L):
    """Packs mu and L into a parameter vector"""
    return np.concatenate((mu, L.flatten()))


def clip(params):
    """Makes sure the Cholensky factor L is well-defined"""
    mu, L = unpack(params)
    # Clip the diagonal to 0.01
    mask = np.logical_and(L < 0.0001, np.eye(K, dtype=bool))
    L[mask] = 0.0001
    # Make sure L is lower triangular
    L[np.triu_indices(K, 1)] = 0
    return pack(mu, L)


def sample_posterior(params):
    """Samples a Q function from the posterior distribution"""
    mu, L = unpack(params)
    return np.dot(L, np.random.randn(K,)) + mu


def objective(samples, params, Q, mu_bar, Sigma_bar_inv):
    """Computes the negative ELBO"""
    mu, L = unpack(params)
    assert mu.shape == (K,) and L.shape == (K,K)
    # We add a small constant to make sure Sigma is always positive definite
    Sigma = np.dot(L, L.T) + np.eye(K) * 0.01
    assert Sigma.shape == (K,K) and np.all(np.linalg.eigvals(Sigma) > 0)
    weights, _ = utils.sample_mvn(n_weights, mu, L)
    assert weights.shape == (n_weights,K)
    likelihood = expected_bellman_error(samples, weights, Q)
    assert likelihood >= 0
    kl = utils.KL(mu, Sigma, mu_bar, Sigma_bar_inv)
    assert kl >= 0
    return likelihood + lambda_ * kl / samples.shape[0]


def expected_bellman_error(samples, weights, Q):
    """Approximates the expected Bellman error with a finite sample of weights"""
    br = bellman_residual(samples, weights, Q) ** 2
    assert br.shape == (samples.shape[0],weights.shape[0])
    errors = np.average(br, axis=0, weights=utils.softmax(br, tau, axis=0))
    assert errors.shape == (weights.shape[0],)
    return np.average(errors)


def bellman_residual(samples, weights, Q):
    """Computes the Bellman residuals of a set of samples given a set of weights"""
    _, _, _, r, s_prime, absorbing, sa = utils.split_data(samples, state_dim, action_dim)
    feats_s_prime = Q.compute_gradient_all_actions(s_prime)
    assert feats_s_prime.shape == (samples.shape[0], n_actions, K)
    feats = Q.compute_gradient(sa)
    assert feats.shape == (samples.shape[0], K)
    Q_values = np.dot(feats, weights.T)
    assert Q_values.shape == (samples.shape[0], weights.shape[0])
    Q_values_prime = np.dot(feats_s_prime, weights.T)
    assert Q_values_prime.shape == (samples.shape[0], n_actions, weights.shape[0])
    mm = utils.mellow_max(Q_values_prime, kappa, axis=1)
    assert mm.shape == (samples.shape[0], weights.shape[0])

    return r[:, np.newaxis] + gamma * mm * (1 - absorbing[:, np.newaxis]) - Q_values


def gradient(samples, params, Q, mu_bar, Sigma_bar_inv):
    """Computes the objective function gradient"""
    mu, L = unpack(params)
    assert mu.shape == (K,) and L.shape == (K,K)
    ws, vs = utils.sample_mvn(n_weights, mu, L)
    assert vs.shape == (n_weights, K) and ws.shape == (n_weights,K)
    be_grad = gradient_be(samples, ws, Q)
    assert be_grad.shape == (n_weights, K)
    # Gradient of the expected Bellman error wrt mu
    ebe_grad_mu = np.average(be_grad, axis=0)
    assert ebe_grad_mu.shape == (K,)
    # Gradient of the expected Bellman error wrt L. TODO is this one correct?
    ebe_grad_L = np.average(be_grad[:, :, np.newaxis] * vs[:, np.newaxis, :], axis=0)
    assert ebe_grad_L.shape == (K,K)
    kl_grad_mu, kl_grad_L = utils.gradient_KL(mu, L, mu_bar, Sigma_bar_inv)
    assert kl_grad_mu.shape == (K,) and kl_grad_L.shape == (K,K)
    grad_mu = ebe_grad_mu + lambda_ * kl_grad_mu / samples.shape[0]
    assert grad_mu.shape == (K,)
    grad_L = ebe_grad_L + lambda_ * kl_grad_L / samples.shape[0]
    assert grad_L.shape == (K,K)

    return pack(grad_mu, grad_L)


def gradient_be(samples, weights, Q):
    """Computes the gradient of the Bellman error for different weights"""
    br = bellman_residual(samples, weights, Q)
    assert br.shape == (samples.shape[0], weights.shape[0])
    mm_grad = gradient_mm(samples, weights, Q)
    assert mm_grad.shape == (samples.shape[0], weights.shape[0], K)
    q_grad = Q.compute_gradient(samples[:, 1:1+state_dim+action_dim])
    assert q_grad.shape == (samples.shape[0], K)
    res_grad = xi * gamma * mm_grad - q_grad[:, np.newaxis, :]
    assert res_grad.shape == (samples.shape[0], weights.shape[0], K)
    be_grad = 2 * np.sum(br[:, :, np.newaxis] * res_grad * utils.softmax(br ** 2, tau, axis=0)[:, :, np.newaxis], axis=0)
    assert be_grad.shape == (weights.shape[0], K)

    return be_grad


def gradient_mm(samples, weights, Q):
    """Computes the mellowmax gradient for different weights"""
    _, _, _, _, s_prime, absorbing, _ = utils.split_data(samples, state_dim, action_dim)
    # We zero-out features corresponding to terminal states so that their value and their gradient are zero
    feats_s_prime = Q.compute_gradient_all_actions(s_prime) * (1 - absorbing)[:, np.newaxis, np.newaxis]
    assert feats_s_prime.shape == (samples.shape[0], n_actions, K)
    Q_values_prime = np.dot(feats_s_prime, weights.T)
    assert Q_values_prime.shape == (samples.shape[0], n_actions, weights.shape[0])
    sft_Q = utils.softmax(Q_values_prime, kappa, axis=1)
    assert sft_Q.shape == (samples.shape[0], n_actions, weights.shape[0])
    mm_grad = np.squeeze(np.sum(sft_Q[:, :, :, np.newaxis] * feats_s_prime[:, :, np.newaxis, :], axis=1))
    assert mm_grad.shape == (samples.shape[0], weights.shape[0], K)

    return mm_grad


def run(door_x, seed=None):

    if seed is not None:
        np.random.seed(seed)

    # Build the features
    x = np.linspace(0, gw_size, n_basis)
    y = np.linspace(0, gw_size, n_basis)
    a = np.linspace(0, n_actions - 1, n_actions)
    mean_x, mean_y, mean_a = np.meshgrid(x, y, a)
    mean = np.hstack((mean_x.reshape(K, 1), mean_y.reshape(K, 1), mean_a.reshape(K, 1)))
    assert mean.shape == (K, 3)

    state_var = (gw_size / (n_basis - 1) / 3) ** 2
    action_var = 0.01 ** 2
    covar = np.eye(state_dim + action_dim)
    covar[0:state_dim, 0:state_dim] *= state_var
    covar[-1, -1] *= action_var
    assert covar.shape == (3, 3)
    covar = np.tile(covar, (K, 1))
    assert covar.shape == (3 * K, 3)

    features = AGaussianRBF(mean, covar, K=K, dims=state_dim + action_dim)

    # Create Target task
    mdp = WalledGridworld(np.array([gw_size, gw_size]), door_x=door_x)
    Q = LinearQRegressor(features, np.arange(n_actions), state_dim, action_dim)

    # Load weights and construct prior distribution
    weights = utils.load_object(source_file)
    ws = np.array([w[1] for w in weights])
    np.random.shuffle(ws)
    # Take only the first n_source weights
    ws = ws[:n_source, :]
    mu_bar = np.mean(ws, axis=0)
    Sigma_bar = np.cov(ws.T)
    # We use higher regularization for the prior to prevent the ELBO from diverging
    Sigma_bar_inv = np.linalg.inv(Sigma_bar + np.eye(K) * 0.01)
    # We initialize the parameters at the prior with smaller regularization (just to make sure Sigma_bar is pd)
    params = pack(mu_bar, np.linalg.cholesky(Sigma_bar + np.eye(K) * 0.0001))

    # Initialize policies
    pi_u = eGreedyPolicy(Q, Q.actions, epsilon=1)
    pi_g = eGreedyPolicy(Q, Q.actions, epsilon=0)

    # Add a first sample
    dataset = utils.generate_episodes(mdp, pi_u, n_episodes=1, render=False)

    # Results
    iterations = []
    n_samples = []
    rewards = []
    l_2 = []
    l_inf = []
    sft = []
    fvals = []

    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0

    # Learning
    for i in range(max_iter):

        s = mdp.reset()
        h = 0
        # Sample parameters from the posterior distribution
        Q._w = sample_posterior(params)
        while h < mdp.horizon:
            # If we do not use time coherent exploration, resample parameters
            Q._w = sample_posterior(params) if not time_coherent else Q._w
            # Take greedy action wrt current Q-function
            a = np.argmax(Q.compute_all_actions(s))
            # Step
            s_prime, r, done, _ = mdp.step(a)
            # Build the new sample and add it to the dataset
            sample = np.concatenate([np.array([h]), s, np.array([a]), np.array([r]), s_prime, np.array([1 if done else 0])])[np.newaxis, :]
            dataset = np.concatenate((dataset,sample), axis=0)

            # Take n_fit steps of gradient
            for _ in range(n_fit):
                # Shuffle the dataset
                np.random.shuffle(dataset)
                # Estimate gradient
                g = gradient(dataset[:gradient_batch, :], params, Q, mu_bar, Sigma_bar_inv)
                # Take a gradient step
                params, t, m_t, v_t = utils.adam(params, g, t, m_t, v_t, alpha=alpha)
                # Clip parameters
                params = clip(params)

            s = s_prime
            h += 1
            if done:
                break

        # Evaluate MAP Q-function
        mu, _ = unpack(params)
        Q._w = mu
        #utils.plot_Q(Q)
        rew = utils.evaluate_policy(mdp, pi_g, render=render, initial_states=[np.array([0., 0.]) for _ in range(10)])
        br = np.squeeze(bellman_residual(dataset, mu[np.newaxis, :], Q) ** 2)
        l_2_err = np.average(br)
        l_inf_err = np.max(br)
        sft_err = np.sum(utils.softmax(br, tau) * br)
        fval = objective(dataset, params, Q, mu_bar, Sigma_bar_inv)

        # Append results
        iterations.append(i)
        n_samples.append(dataset.shape[0])
        rewards.append(rew)
        l_2.append(l_2_err)
        l_inf.append(l_inf_err)
        sft.append(sft_err)
        fvals.append(fval)

        if verbose:
            print("Iteration {} Reward {} Fval {} L2 {} L_inf {} Sft {}".format(i,rew[0],fval,l_2_err,l_inf_err,sft_err))

    run_info = [iterations, n_samples, rewards, l_2, l_inf, sft, fval]
    weights = np.array(mu)

    return [door_x, weights, run_info]


# Global parameters
gamma = 0.99
n_actions = 4
state_dim = 2
action_dim = 1
render = False
verbose = True

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--kappa", default=100.)
parser.add_argument("--xi", default=1.0)
parser.add_argument("--tau", default=0.0)
parser.add_argument("--gradient_batch", default=1000)
parser.add_argument("--max_iter", default=100)
parser.add_argument("--n_fit", default=1)
parser.add_argument("--alpha", default=0.001)
parser.add_argument("--lambda_", default=0.001)
parser.add_argument("--time_coherent", default=False)
parser.add_argument("--n_weights", default=100)
parser.add_argument("--n_source", default=10)
parser.add_argument("--gw_size", default=5)
# Door at -1 means random positions over all runs
parser.add_argument("--door", default=-1)
parser.add_argument("--n_basis", default=6)
parser.add_argument("--n_jobs", default=1)
parser.add_argument("--n_runs", default=1)
parser.add_argument("--file_name", default="gvt_{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
parser.add_argument("--source_file", default="../scripts/mm_sources_5x5_basis6_rw1_kappa100_xi1_tau0")

# Read arguments
args = parser.parse_args()
kappa = float(args.kappa)
xi = float(args.xi)
tau = float(args.tau)
gradient_batch = int(args.gradient_batch)
max_iter = int(args.max_iter)
n_fit = int(args.n_fit)
alpha = float(args.alpha)
lambda_ = float(args.lambda_)
time_coherent = bool(args.time_coherent)
n_weights = int(args.n_weights)
n_source = int(args.n_source)
gw_size = int(args.gw_size)
door = float(args.door)
n_basis = int(args.n_basis)
n_jobs = int(args.n_jobs)
n_runs = int(args.n_runs)
file_name = str(args.file_name)
source_file = str(args.source_file)

# Number of features
K = n_basis ** 2 * n_actions

# Generate door positions
doors = [np.random.uniform(0.5, gw_size - 0.5) if door < 0 else door for _ in range(n_runs)]

if n_jobs == 1:
    results = [run(d) for d in doors]
elif n_jobs > 1:
    seeds = [np.random.randint(1000000) for _ in range(n_runs)]
    results = Parallel(n_jobs=n_jobs)(delayed(run)(d,seed) for (d,seed) in zip(doors,seeds))

utils.save_object(results, file_name)


