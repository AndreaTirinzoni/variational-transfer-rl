import sys
sys.path.append("../")

import numpy as np
from envs.cartpole import CartPoleEnv
from approximators.mlp import MLPQFunction
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
    # We add a small constant to make sure Sigma is always positive definite
    Sigma = np.dot(L, L.T)
    weights, _ = utils.sample_mvn(n_weights, mu, L)
    likelihood = operator.expected_bellman_error(Q, samples, weights)
    assert likelihood >= 0
    kl = utils.KL(mu, Sigma, mu_bar, Sigma_bar_inv)
    assert kl >= 0
    return likelihood + lambda_ * kl / n_samples


def gradient(samples, params, Q, mu_bar, Sigma_bar_inv, operator, n_samples):
    """Computes the objective function gradient"""
    mu, L = unpack(params)
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


def run(mdp, seed=None):

    if seed is not None:
        np.random.seed(seed)

    # Create BellmanOperator
    operator = MellowBellmanOperator(kappa, tau, xi, gamma, state_dim, action_dim)
    # Create Q Function
    layers = [l1]
    if l2 > 0:
        layers.append(l2)
    Q = MLPQFunction(state_dim, n_actions, layers=layers)
    # Set number of weights
    global K
    K = Q._nn.num_weights
    # Initialize policies
    pi_u = EpsilonGreedy(Q, np.arange(n_actions), epsilon=1)
    pi_g = EpsilonGreedy(Q, np.arange(n_actions), epsilon=0)

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

    # Create masks for ADAM and SGD
    adam_mask = pack(np.ones(K) * alpha_adam, np.zeros((K,K)))  # ADAM learns only \mu
    sgd_mask = pack(np.zeros(K), np.ones((K,K)) * alpha_sgd)  # SGD learns only L

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
        a = np.argmax(Q.value_actions(s))
        # Step
        s_prime, r, done, _ = mdp.step(a)
        # Build the new sample and add it to the dataset
        dataset = utils.add_sample(dataset, buffer_size, h, s, np.array([a]), r, s_prime, done)

        # Take a step of gradient if needed
        if i % train_freq == 0:
            # Shuffle the dataset
            np.random.shuffle(dataset)
            # Estimate gradient
            g = gradient(dataset[:batch_size, :], params, Q, mu_bar, Sigma_bar_inv, operator, n_init_samples + i + 1)
            # Take a gradient step for \mu
            params, t, m_t, v_t = utils.adam(params, g, t, m_t, v_t, alpha=adam_mask)
            # Take a gradient step for L
            params = utils.sgd(params, g, alpha=sgd_mask)
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
            rew = utils.evaluate_policy(mdp, pi_g, render=render, n_episodes=n_eval_episodes)[0]
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

    run_info = [iterations, episodes, n_samples, learning_rewards, evaluation_rewards, l_2, l_inf, sft, fvals]
    weights = np.array(mu)

    return [(mdp.masscart,mdp.masspole,mdp.length), weights, run_info]


# Global parameters
gamma = 0.99
n_actions = 2
state_dim = 4
action_dim = 1
render = False
verbose = True

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--kappa", default=100.)
parser.add_argument("--xi", default=0.5)
parser.add_argument("--tau", default=0.0)
parser.add_argument("--batch_size", default=50)
parser.add_argument("--max_iter", default=5000)
parser.add_argument("--buffer_size", default=10000)
parser.add_argument("--random_episodes", default=0)
parser.add_argument("--train_freq", default=1)
parser.add_argument("--eval_freq", default=100)
parser.add_argument("--mean_episodes", default=20)
parser.add_argument("--l1", default=32)
parser.add_argument("--l2", default=0)
parser.add_argument("--alpha_adam", default=0.001)
parser.add_argument("--alpha_sgd", default=0.1)
parser.add_argument("--lambda_", default=0.001)
parser.add_argument("--time_coherent", default=False)
parser.add_argument("--n_weights", default=10)
parser.add_argument("--n_source", default=10)
parser.add_argument("--sigma_reg", default=0.0001)
parser.add_argument("--cholesky_clip", default=0.0001)
parser.add_argument("--env", default="cartpole")
# Cartpole parameters (default = randomize
parser.add_argument("--cart_mass", default=-1)
parser.add_argument("--pole_mass", default=-1)
parser.add_argument("--pole_length", default=-1)
parser.add_argument("--n_jobs", default=1)
parser.add_argument("--n_runs", default=1)
parser.add_argument("--file_name", default="gvt_{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
parser.add_argument("--source_file", default="source_tasks/cartpole_nn32")

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
l1 = int(args.l1)
l2 = int(args.l2)
alpha_adam = float(args.alpha_adam)
alpha_sgd = float(args.alpha_sgd)
lambda_ = float(args.lambda_)
time_coherent = bool(args.time_coherent)
n_weights = int(args.n_weights)
n_source = int(args.n_source)
sigma_reg = float(args.sigma_reg)
cholesky_clip = float(args.cholesky_clip)
env = str(args.env)
cart_mass = float(args.cart_mass)
pole_mass = float(args.pole_mass)
pole_length = float(args.pole_length)
n_jobs = int(args.n_jobs)
n_runs = int(args.n_runs)
file_name = str(args.file_name)
source_file = str(args.source_file)

# Number of weights
K = 0

# Generate tasks
mc = [np.random.uniform(0.5, 1.5) if cart_mass < 0 else cart_mass for _ in range(n_runs)]
mp = [np.random.uniform(0.1, 0.2) if pole_mass < 0 else pole_mass for _ in range(n_runs)]
l = [np.random.uniform(0.2, 0.8) if pole_length < 0 else pole_length for _ in range(n_runs)]
mdps = [CartPoleEnv(a,b,c) for a,b,c in zip(mc,mp,l)]
n_eval_episodes = 5

if n_jobs == 1:
    results = [run(mdp) for mdp in mdps]
elif n_jobs > 1:
    seeds = [np.random.randint(1000000) for _ in range(n_runs)]
    results = Parallel(n_jobs=n_jobs)(delayed(run)(mdp,seed) for (mdp,seed) in zip(mdps,seeds))

utils.save_object(results, file_name)


