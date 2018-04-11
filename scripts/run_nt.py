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


def gradient(Q, samples):
    """Computes the Bellman error gradient"""
    _, _, _, _, s_prime, absorbing, sa = utils.split_data(samples, state_dim, action_dim)
    br = bellman_residual(Q, samples)
    assert br.shape == (samples.shape[0],)
    mm_gradient = gradient_mm(Q, s_prime, absorbing)
    assert mm_gradient.shape == (samples.shape[0], K)
    q_gradient = Q.compute_gradient(sa)
    assert q_gradient.shape == (samples.shape[0], K)
    b_grad = xi * gamma * mm_gradient - q_gradient
    assert b_grad.shape == (samples.shape[0], K)
    bellman_grad = 2 * np.sum(br[:, np.newaxis] * b_grad * utils.softmax(br ** 2, tau)[:, np.newaxis], axis=0)
    assert bellman_grad.shape == (K,)

    return bellman_grad


def gradient_mm(Q, s_prime, absorbing):
    """Computes the mellow-max gradient"""
    q_values = Q.compute_all_actions(s_prime, absorbing)
    assert q_values.shape == (s_prime.shape[0],n_actions)
    q_gradient = Q.compute_gradient_all_actions(s_prime) * (1 - absorbing)[:, np.newaxis, np.newaxis]
    assert q_gradient.shape == (s_prime.shape[0],n_actions,K)
    sft_Q = utils.softmax(q_values, kappa, axis=1)
    assert sft_Q.shape == (s_prime.shape[0], n_actions)
    mm_grad = np.sum(sft_Q[:, :, np.newaxis] * q_gradient, axis=1)
    assert mm_grad.shape == (s_prime.shape[0], K)

    return mm_grad


def bellman_residual(Q, samples):
    """Computes the Bellman residuals of given samples"""
    _, _, _, r, s_prime, absorbing, sa = utils.split_data(samples, state_dim, action_dim)
    Qs = Q(sa)
    assert Qs.shape == (samples.shape[0],)
    Qs_prime = Q.compute_all_actions(s_prime, absorbing)
    assert Qs_prime.shape == (samples.shape[0],n_actions)
    mmQs = utils.mellow_max(Qs_prime, kappa)
    assert mmQs.shape == (samples.shape[0],)

    return r + gamma * mmQs * (1 - absorbing) - Qs


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

    # Initialize policies
    pi = eGreedyPolicy(Q, Q.actions, epsilon=epsilon)
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

    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0

    # Learning
    for i in range(max_iter):

        s = mdp.reset()
        h = 0
        while h < mdp.horizon:
            # Take epsilon-greedy action wrt current Q-function
            a = pi.sample_action(s)
            # Step
            s_prime, r, done, _ = mdp.step(a)
            # Build the new sample and add it to the dataset
            sample = np.concatenate([np.array([h]), s, a, np.array([r]), s_prime, np.array([1 if done else 0])])[np.newaxis, :]
            dataset = np.concatenate((dataset,sample), axis=0)

            # Take n_fit steps of gradient
            for _ in range(n_fit):
                # Shuffle the dataset
                np.random.shuffle(dataset)
                # Estimate gradient
                g = gradient(Q, dataset[:gradient_batch, :])
                # Take a gradient step
                Q._w, t, m_t, v_t = utils.adam(Q._w, g, t, m_t, v_t, alpha=alpha)

            s = s_prime
            h += 1
            if done:
                break

        # Evaluate MAP Q-function
        #utils.plot_Q(Q)
        rew = utils.evaluate_policy(mdp, pi_g, render=render, initial_states=[np.array([0., 0.]) for _ in range(10)])
        br = bellman_residual(Q, dataset) ** 2
        l_2_err = np.average(br)
        l_inf_err = np.max(br)
        sft_err = np.sum(utils.softmax(br, tau) * br)

        # Append results
        iterations.append(i)
        n_samples.append(dataset.shape[0])
        rewards.append(rew)
        l_2.append(l_2_err)
        l_inf.append(l_inf_err)
        sft.append(sft_err)

        if verbose:
            print("Iteration {} Reward {} L2 {} L_inf {} Sft {}".format(i,rew[0],l_2_err,l_inf_err,sft_err))

    run_info = [iterations, n_samples, rewards, l_2, l_inf, sft]
    weights = np.array(Q._w)

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
parser.add_argument("--epsilon", default=0.2)
parser.add_argument("--gradient_batch", default=100)
parser.add_argument("--max_iter", default=100)
parser.add_argument("--n_fit", default=1)
parser.add_argument("--alpha", default=0.001)
parser.add_argument("--gw_size", default=5)
# Door at -1 means random positions over all runs
parser.add_argument("--door", default=-1)
parser.add_argument("--n_basis", default=6)
parser.add_argument("--n_jobs", default=1)
parser.add_argument("--n_runs", default=1)
parser.add_argument("--file_name", default="gvt_{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))

# Read arguments
args = parser.parse_args()
kappa = float(args.kappa)
xi = float(args.xi)
tau = float(args.tau)
epsilon = float(args.epsilon)
gradient_batch = int(args.gradient_batch)
max_iter = int(args.max_iter)
n_fit = int(args.n_fit)
alpha = float(args.alpha)
gw_size = int(args.gw_size)
door = float(args.door)
n_basis = int(args.n_basis)
n_jobs = int(args.n_jobs)
n_runs = int(args.n_runs)
file_name = str(args.file_name)

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


