import sys
sys.path.append("../")

import numpy as np
from envs.walled_gridworld import WalledGridworld
from features.agrbf import build_features_gw
from approximators.linear import LinearQFunction
from operators.mellow import MellowBellmanOperator
from policies import EpsilonGreedy
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


def objective(samples, params, Q, mu_bar, Sigma_bar_inv, operator, n_samples):
    """Computes the negative ELBO"""
    mu, L = unpack(params)
    assert mu.shape == (K,) and L.shape == (K,K)
    # We add a small constant to make sure Sigma is always positive definite
    Sigma = np.dot(L, L.T) + np.eye(K) * 0.01
    assert Sigma.shape == (K,K) and np.all(np.linalg.eigvals(Sigma) > 0)
    weights, _ = utils.sample_mvn(n_weights, mu, L)
    assert weights.shape == (n_weights,K)
    likelihood = operator.expected_bellman_error(Q, samples, weights)
    assert likelihood >= 0
    kl = utils.KL(mu, Sigma, mu_bar, Sigma_bar_inv)
    assert kl >= 0
    return likelihood + lambda_ * kl / n_samples


def gradient(samples, params, Q, mu_bar, Sigma_bar_inv, operator, n_samples):
    """Computes the objective function gradient"""
    mu, L = unpack(params)
    assert mu.shape == (K,) and L.shape == (K,K)
    ws, vs = utils.sample_mvn(n_weights, mu, L)
    assert vs.shape == (n_weights, K) and ws.shape == (n_weights,K)
    be_grad = operator.gradient_be(Q, samples, ws)
    assert be_grad.shape == (n_weights, K)
    # Gradient of the expected Bellman error wrt mu
    ebe_grad_mu = np.average(be_grad, axis=0)
    assert ebe_grad_mu.shape == (K,)
    # Gradient of the expected Bellman error wrt L. TODO is this one correct?
    ebe_grad_L = np.average(be_grad[:, :, np.newaxis] * vs[:, np.newaxis, :], axis=0)
    assert ebe_grad_L.shape == (K,K)
    kl_grad_mu, kl_grad_L = utils.gradient_KL(mu, L, mu_bar, Sigma_bar_inv)
    assert kl_grad_mu.shape == (K,) and kl_grad_L.shape == (K,K)
    grad_mu = ebe_grad_mu + lambda_ * kl_grad_mu / n_samples
    assert grad_mu.shape == (K,)
    grad_L = ebe_grad_L + lambda_ * kl_grad_L / n_samples
    assert grad_L.shape == (K,K)

    return pack(grad_mu, grad_L)


def run(mdp, seed=None):

    if seed is not None:
        np.random.seed(seed)

    # Build the features
    features = build_features_gw(gw_size, n_basis, n_actions, state_dim, action_dim)

    # Create BellmanOperator
    operator = MellowBellmanOperator(kappa, tau, xi, gamma, state_dim, action_dim)

    # Create Q Function
    Q = LinearQFunction(features, np.arange(n_actions), state_dim, action_dim)

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
    pi_u = EpsilonGreedy(Q, Q.actions, epsilon=1)
    pi_g = EpsilonGreedy(Q, Q.actions, epsilon=0)

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
            a = np.argmax(Q.value_actions(s))
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
                g = gradient(dataset[:gradient_batch, :], params, Q, mu_bar, Sigma_bar_inv, operator, dataset.shape[0])
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
        br = np.squeeze(operator.bellman_residual(Q, dataset) ** 2)
        l_2_err = np.average(br)
        l_inf_err = np.max(br)
        sft_err = np.sum(utils.softmax(br, tau) * br)
        fval = objective(dataset, params, Q, mu_bar, Sigma_bar_inv, operator, dataset.shape[0])

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

    return [mdp.door_x, weights, run_info]


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
parser.add_argument("--xi", default=0.5)
parser.add_argument("--tau", default=0.0)
parser.add_argument("--gradient_batch", default=100)
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

# Generate tasks
doors = [np.random.uniform(0.5, gw_size - 0.5) if door < 0 else door for _ in range(n_runs)]
mdps = [WalledGridworld(np.array([gw_size, gw_size]), door_x=d) for d in doors]

if n_jobs == 1:
    results = [run(mdp) for mdp in mdps]
elif n_jobs > 1:
    seeds = [np.random.randint(1000000) for _ in range(n_runs)]
    results = Parallel(n_jobs=n_jobs)(delayed(run)(mdp,seed) for (mdp,seed) in zip(mdps,seeds))

utils.save_object(results, file_name)


