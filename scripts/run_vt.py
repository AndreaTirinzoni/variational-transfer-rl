import sys
sys.path.append("../")

import numpy as np
from envs.walled_gridworld import WalledGridworld
from envs.marcellos_gridworld import MarcellosGridworld
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
    mask = np.logical_and(L < cholesky_clip, np.eye(K, dtype=bool))
    L[mask] = cholesky_clip
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
    Sigma = np.dot(L, L.T) + np.eye(K) * sigma_reg
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
    #assert kl_grad_mu.shape == (K,) and kl_grad_L.shape == (K,K)
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
    # Initialize policies
    pi_u = EpsilonGreedy(Q, Q.actions, epsilon=1)
    pi_g = EpsilonGreedy(Q, Q.actions, epsilon=0)

    # Add random episodes if needed
    dataset = utils.generate_episodes(mdp, pi_u, n_episodes=random_episodes) if random_episodes > 0 else None
    n_init_samples = dataset.shape[0] if dataset is not None else 0

    # Load weights and construct prior distribution
    weights = utils.load_object(source_file)
    ws = np.array([w[1] for w in weights])
    np.random.shuffle(ws)
    # Take only the first n_source weights
    ws = ws[:n_source, :]
    mu_bar = np.mean(ws, axis=0)
    Sigma_bar = np.cov(ws.T)
    # We use higher regularization for the prior to prevent the ELBO from diverging
    Sigma_bar_inv = np.linalg.inv(Sigma_bar + np.eye(K) * sigma_reg)
    # We initialize the parameters at the prior with smaller regularization (just to make sure Sigma_bar is pd)
    params = clip(pack(mu_bar, np.linalg.cholesky(Sigma_bar + np.eye(K) * cholesky_clip)))

    # Results
    iterations = []
    episodes = []
    n_samples = []
    evaluation_rewards = []
    learning_rewards = []
    episode_rewards = [0.0]
    l_2 = []
    l_inf = []
    sft = []
    fvals = []

    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0

    # Init env
    s = mdp.reset()
    h = 0
    Q._w = sample_posterior(params)

    # Learning
    for i in range(max_iter):

        # If we do not use time coherent exploration, resample parameters
        Q._w = sample_posterior(params) if not time_coherent else Q._w
        # Take greedy action wrt current Q-function
        a = np.array([np.argmax(Q.value_actions(s))])
        # Step
        s_prime, r, done, _ = mdp.step(a)
        # Build the new sample and add it to the dataset
        dataset = utils.add_sample(dataset, buffer_size, h, s, a, r, s_prime, done)

        # Take a step of gradient if needed
        if i % train_freq == 0:
            # Shuffle the dataset
            np.random.shuffle(dataset)
            # Estimate gradient
            g = gradient(dataset[:batch_size, :], params, Q, mu_bar, Sigma_bar_inv, operator, n_init_samples + i + 1)
            # Take a gradient step
            params, t, m_t, v_t = utils.adam(params, g, t, m_t, v_t, alpha=alpha)
            # Clip parameters
            params = clip(params)

        # Add reward to last episode
        episode_rewards[-1] += r * gamma ** h

        s = s_prime
        h += 1
        if done or h >= mdp.horizon:

            episode_rewards.append(0.0)
            s = mdp.reset()
            h = 0
            Q._w = sample_posterior(params)

        # Evaluate model
        if i % eval_freq == 0:

            #Save current weights
            current_w = np.array(Q._w)

            # Evaluate MAP Q-function
            mu, _ = unpack(params)
            Q._w = mu
            # utils.plot_Q(Q)
            rew = utils.evaluate_policy(mdp, pi_g, render=render, initial_states=eval_states)[0]
            learning_rew = np.mean(episode_rewards[-mean_episodes - 1:-1]) if len(episode_rewards) > 1 else 0.0
            br = operator.bellman_residual(Q, dataset) ** 2
            l_2_err = np.average(br)
            l_inf_err = np.max(br)
            sft_err = np.sum(utils.softmax(br, tau) * br)
            fval = objective(dataset, params, Q, mu_bar, Sigma_bar_inv, operator, n_init_samples + i + 1)

            # Append results
            iterations.append(i)
            episodes.append(len(episode_rewards) - 1)
            n_samples.append(n_init_samples + i + 1)
            evaluation_rewards.append(rew)
            learning_rewards.append(learning_rew)
            l_2.append(l_2_err)
            l_inf.append(l_inf_err)
            sft.append(sft_err)
            fvals.append(fval)

            # Make sure we restart from s
            mdp.reset(s)

            # Restore weights
            Q._w = current_w

            if verbose:
                print("Iter {} Episodes {} Rew(G) {} Rew(L) {} Fval {} L2 {} L_inf {} Sft {}".format(
                    i, episodes[-1], rew, learning_rew, fval, l_2_err, l_inf_err, sft_err))

    run_info = [iterations, episodes, n_samples, learning_rewards, evaluation_rewards, l_2, l_inf, sft, fval]
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
parser.add_argument("--batch_size", default=100)
parser.add_argument("--max_iter", default=5000)
parser.add_argument("--buffer_size", default=10000)
parser.add_argument("--random_episodes", default=0)
parser.add_argument("--train_freq", default=1)
parser.add_argument("--eval_freq", default=50)
parser.add_argument("--mean_episodes", default=50)
parser.add_argument("--alpha", default=0.001)
parser.add_argument("--lambda_", default=0.001)
parser.add_argument("--time_coherent", default=False)
parser.add_argument("--n_weights", default=100)
parser.add_argument("--n_source", default=10)
parser.add_argument("--sigma_reg", default=0.01)
parser.add_argument("--cholesky_clip", default=0.0001)
parser.add_argument("--env", default="two-room-gw")
parser.add_argument("--gw_size", default=5)
# Door at -1 means random positions over all runs
parser.add_argument("--door", default=-1)
parser.add_argument("--door2", default=-1)
parser.add_argument("--n_basis", default=6)
parser.add_argument("--n_jobs", default=1)
parser.add_argument("--n_runs", default=1)
parser.add_argument("--file_name", default="gvt_{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
parser.add_argument("--source_file", default="source_tasks/gw5x5")

# Read arguments
args = parser.parse_args()
kappa = float(args.kappa)
xi = float(args.xi)
tau = float(args.tau)
batch_size = int(args.batch_size)
max_iter = int(args.max_iter)
buffer_size = int(args.buffer_size)
random_episodes = int(args.random_episodes)
train_freq = int(args.train_freq)
eval_freq = int(args.eval_freq)
mean_episodes = int(args.mean_episodes)
alpha = float(args.alpha)
lambda_ = float(args.lambda_)
time_coherent = bool(args.time_coherent)
n_weights = int(args.n_weights)
n_source = int(args.n_source)
sigma_reg = float(args.sigma_reg)
cholesky_clip = float(args.cholesky_clip)
env = str(args.env)
gw_size = int(args.gw_size)
door = float(args.door)
door2 = float(args.door2)
n_basis = int(args.n_basis)
n_jobs = int(args.n_jobs)
n_runs = int(args.n_runs)
file_name = str(args.file_name)
source_file = str(args.source_file)

# Number of features
K = n_basis ** 2 * n_actions

# Generate tasks
doors = [np.random.uniform(0.5, gw_size - 0.5) if door < 0 else door for _ in range(n_runs)]
doors2 = [np.random.uniform(0.5, gw_size - 0.5) if door2 < 0 else door2 for _ in range(n_runs)]
if env == "two-room-gw":
    mdps = [WalledGridworld(np.array([gw_size, gw_size]), door_x=d) for d in doors]
elif env == "three-room-gw":
    mdps = [MarcellosGridworld(np.array([gw_size, gw_size]), door_x=(d1,d2)) for (d1,d2) in zip(doors,doors2)]
eval_states = [np.array([0., 0.]) for _ in range(10)]

if n_jobs == 1:
    results = [run(mdp) for mdp in mdps]
elif n_jobs > 1:
    seeds = [np.random.randint(1000000) for _ in range(n_runs)]
    results = Parallel(n_jobs=n_jobs)(delayed(run)(mdp,seed) for (mdp,seed) in zip(mdps,seeds))

utils.save_object(results, file_name)


