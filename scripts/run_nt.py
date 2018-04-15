import sys
sys.path.append("../")

import numpy as np
from envs.walled_gridworld import WalledGridworld
from envs.marcellos_gridworld import MarcellosGridworld
from features.agrbf import build_features_gw
from approximators.linear import LinearQFunction
from operators.mellow import MellowBellmanOperator
from policies import EpsilonGreedy, ScheduledEpsilonGreedy
import utils
import argparse
from joblib import Parallel, delayed
import datetime


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
    schedule = np.linspace(eps_start,eps_end,exploration_fraction*max_iter)
    pi = ScheduledEpsilonGreedy(Q, Q.actions, schedule)
    pi_u = EpsilonGreedy(Q, Q.actions, epsilon=1)
    pi_g = EpsilonGreedy(Q, Q.actions, epsilon=0)

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
        dataset = utils.add_sample(dataset, buffer_size, h, s, a, r, s_prime, done)

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
            rew = utils.evaluate_policy(mdp, pi_g, render=render, initial_states=eval_states)[0]
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
parser.add_argument("--exploration_fraction", default=0.2)
parser.add_argument("--eps_start", default=1.0)
parser.add_argument("--eps_end", default=0.02)
parser.add_argument("--train_freq", default=1)
parser.add_argument("--eval_freq", default=50)
parser.add_argument("--mean_episodes", default=50)
parser.add_argument("--alpha", default=0.001)
parser.add_argument("--env", default="two-room-gw")
parser.add_argument("--gw_size", default=5)
# Door at -1 means random positions over all runs
parser.add_argument("--door", default=-1)
parser.add_argument("--door2", default=-1)
parser.add_argument("--n_basis", default=6)
parser.add_argument("--n_jobs", default=1)
parser.add_argument("--n_runs", default=1)
parser.add_argument("--file_name", default="gvt_{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))

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
alpha = float(args.alpha)
env = str(args.env)
gw_size = int(args.gw_size)
door = float(args.door)
door2 = float(args.door)
n_basis = int(args.n_basis)
n_jobs = int(args.n_jobs)
n_runs = int(args.n_runs)
file_name = str(args.file_name)

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


