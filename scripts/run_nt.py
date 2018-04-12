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


def run(door_x, seed=None):

    if seed is not None:
        np.random.seed(seed)

    # Build the features
    features = build_features_gw(gw_size, n_basis, n_actions, state_dim, action_dim)

    # Create BellmanOperator
    operator = MellowBellmanOperator(kappa, tau, xi, gamma, state_dim, action_dim)

    # Create Target task
    mdp = WalledGridworld(np.array([gw_size, gw_size]), door_x=door_x)
    Q = LinearQFunction(features, np.arange(n_actions), state_dim, action_dim)

    # Initialize policies
    pi = EpsilonGreedy(Q, Q.actions, epsilon=epsilon)
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
                g = operator.gradient_be(Q, dataset[:gradient_batch, :])
                # Take a gradient step
                Q._w, t, m_t, v_t = utils.adam(Q._w, g, t, m_t, v_t, alpha=alpha)

            s = s_prime
            h += 1
            if done:
                break

        # Evaluate MAP Q-function
        #utils.plot_Q(Q)
        rew = utils.evaluate_policy(mdp, pi_g, render=render, initial_states=[np.array([0., 0.]) for _ in range(10)])
        br = operator.bellman_residual(Q, dataset) ** 2
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


