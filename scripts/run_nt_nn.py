import sys
sys.path.append("../")

import numpy as np
from envs.cartpole import CartPoleEnv
from approximators.mlp import MLPQFunction
from operators.mellow import MellowBellmanOperator
from policies import EpsilonGreedy, ScheduledEpsilonGreedy
import utils
import argparse
from joblib import Parallel, delayed
import datetime


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
    # Initialize policies
    schedule = np.linspace(eps_start,eps_end,exploration_fraction*max_iter)
    pi = ScheduledEpsilonGreedy(Q, np.arange(n_actions), schedule)
    pi_u = EpsilonGreedy(Q, np.arange(n_actions), epsilon=1)
    pi_g = EpsilonGreedy(Q, np.arange(n_actions), epsilon=0)

    # Add random episodes if needed
    dataset = utils.generate_episodes(mdp, pi_u, n_episodes=random_episodes) if random_episodes > 0 else None
    n_init_samples = dataset.shape[0] if dataset is not None else 0

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

    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0

    # Init env
    s = mdp.reset()
    h = 0

    # Learning
    for i in range(max_iter):

        # Take epsilon-greedy action wrt current Q-function
        a = pi.sample_action(s)
        # Step
        s_prime, r, done, _ = mdp.step(a)
        # Build the new sample and add it to the dataset
        dataset = utils.add_sample(dataset, buffer_size, h, s, np.array([a]), r, s_prime, done)

        # Take a step of gradient if needed
        if i % train_freq == 0:
            # Shuffle the dataset
            np.random.shuffle(dataset)
            # Estimate gradient
            g = operator.gradient_be(Q, dataset[:batch_size, :])
            # Take a gradient step
            Q._w, t, m_t, v_t = utils.adam(Q._w, g, t, m_t, v_t, alpha=alpha)

        # Add reward to last episode
        episode_rewards[-1] += r * gamma ** h

        s = s_prime
        h += 1
        if done or h >= mdp.horizon:

            episode_rewards.append(0.0)
            s = mdp.reset()
            h = 0

        # Evaluate model
        if i % eval_freq == 0:

            # Evaluate greedy policy
            #utils.plot_Q(Q)
            rew = utils.evaluate_policy(mdp, pi_g, render=render, n_episodes=n_eval_episodes)[0]
            learning_rew = np.mean(episode_rewards[-mean_episodes-1:-1]) if len(episode_rewards) > 1 else 0.0
            br = operator.bellman_residual(Q, dataset) ** 2
            l_2_err = np.average(br)
            l_inf_err = np.max(br)
            sft_err = np.sum(utils.softmax(br, tau) * br)

            # Append results
            iterations.append(i)
            episodes.append(len(episode_rewards) - 1)
            n_samples.append(n_init_samples + i + 1)
            evaluation_rewards.append(rew)
            learning_rewards.append(learning_rew)
            l_2.append(l_2_err)
            l_inf.append(l_inf_err)
            sft.append(sft_err)

            # Make sure we restart from s
            mdp.reset(s)

            if verbose:
                print("Iter {} Episodes {} Rew(G) {} Rew(L) {} L2 {} L_inf {} Sft {}".format(
                    i, episodes[-1], rew, learning_rew, l_2_err, l_inf_err, sft_err))

    run_info = [iterations, episodes, n_samples, learning_rewards, evaluation_rewards, l_2, l_inf, sft]
    weights = np.array(Q._w)

    return [None, weights, run_info]


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
parser.add_argument("--exploration_fraction", default=0.2)
parser.add_argument("--eps_start", default=1.0)
parser.add_argument("--eps_end", default=0.02)
parser.add_argument("--train_freq", default=1)
parser.add_argument("--eval_freq", default=100)
parser.add_argument("--mean_episodes", default=20)
parser.add_argument("--l1", default=64)
parser.add_argument("--l2", default=0)
parser.add_argument("--alpha", default=0.001)
parser.add_argument("--env", default="cartpole")
# Cartpole parameters (default = randomize
parser.add_argument("--cart_mass", default=-1)
parser.add_argument("--pole_mass", default=-1)
parser.add_argument("--pole_length", default=-1)
parser.add_argument("--n_jobs", default=1)
parser.add_argument("--n_runs", default=1)
parser.add_argument("--file_name", default="nt_{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))

# Read arguments
args = parser.parse_args()
kappa = float(args.kappa)
xi = float(args.xi)
tau = float(args.tau)
batch_size = int(args.batch_size)
max_iter = int(args.max_iter)
buffer_size = int(args.buffer_size)
random_episodes = int(args.random_episodes)
exploration_fraction = float(args.exploration_fraction)
eps_start = float(args.eps_start)
eps_end = float(args.eps_end)
train_freq = int(args.train_freq)
eval_freq = int(args.eval_freq)
mean_episodes = int(args.mean_episodes)
l1 = int(args.l1)
l2 = int(args.l2)
alpha = float(args.alpha)
env = str(args.env)
cart_mass = float(args.cart_mass)
pole_mass = float(args.pole_mass)
pole_length = float(args.pole_length)
n_jobs = int(args.n_jobs)
n_runs = int(args.n_runs)
file_name = str(args.file_name)

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


