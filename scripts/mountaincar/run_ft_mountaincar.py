import sys
import os
path = os.path.dirname(os.path.realpath(__file__))  # path to this directory
sys.path.append(os.path.abspath(path + "/../.."))

import numpy as np
from envs.mountain_car import MountainCarEnv
from approximators.mlp_torch import MLPQFunction
from operators.mellow_torch import MellowBellmanOperator
from algorithms.nt import learn
from misc import utils
import argparse
from joblib import Parallel, delayed
import datetime
from algorithms.dqn import DQN


# Global parameters
render = False
verbose = True

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--kappa", default=100.)
parser.add_argument("--xi", default=0.5)
parser.add_argument("--tau", default=0.0)
parser.add_argument("--batch_size", default=32)
parser.add_argument("--max_iter", default=10000)
parser.add_argument("--buffer_size", default=50000)
parser.add_argument("--random_episodes", default=0)
parser.add_argument("--exploration_fraction", default=0.1)
parser.add_argument("--eps_start", default=1.0)
parser.add_argument("--eps_end", default=0.01)
parser.add_argument("--train_freq", default=1)
parser.add_argument("--eval_freq", default=100)
parser.add_argument("--mean_episodes", default=20)
parser.add_argument("--l1", default=64)
parser.add_argument("--l2", default=0)
parser.add_argument("--alpha", default=0.001)
parser.add_argument("--env", default="cartpole")
# Car parameters (default = randomize)
parser.add_argument("--speed", default=-1)
parser.add_argument("--n_jobs", default=1)
parser.add_argument("--n_runs", default=1)
parser.add_argument("--file_name", default="ft_{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
parser.add_argument("--dqn", default=False)
parser.add_argument("--source_file", default="mountaincar_sources_64nn_200k")


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
speed = float(args.speed)
n_jobs = int(args.n_jobs)
n_runs = int(args.n_runs)
file_name = str(args.file_name)
dqn = bool(args.dqn)
source_file = str(args.source_file)

# Seed to get reproducible results
np.random.seed(485)

# Generate tasks
vel = [np.random.uniform(0.001, 0.0015) if speed < 0 else speed for _ in range(n_runs)]
mdps = [MountainCarEnv(vel[i]) for i in range(n_runs)]
n_eval_episodes = 5

state_dim = mdps[0].state_dim
action_dim = 1
n_actions = mdps[0].action_space.n


layers = [l1]
if l2 > 0:
    layers.append(l2)

if not dqn:
    # Create BellmanOperator
    operator = MellowBellmanOperator(kappa, tau, xi, mdps[0].gamma, state_dim, action_dim)
    # Create Q Function
    Q = MLPQFunction(state_dim, n_actions, layers=layers)
else:
    Q, operator = DQN(state_dim, action_dim, n_actions, mdps[0].gamma, layers=layers)

# set init weights
weights = utils.load_object(source_file)
ws = np.array([w[1] for w in weights])
np.random.shuffle(ws)
params = np.array([w[0][1:] for w in weights])

def run(mdp, seed=None, idx=0):
    Q._w = ws[idx]
    return learn(mdp,
                 Q,
                 operator,
                 max_iter=max_iter,
                 buffer_size=buffer_size,
                 batch_size=batch_size,
                 alpha=alpha,
                 train_freq=train_freq,
                 eval_freq=eval_freq,
                 eps_start=eps_start,
                 eps_end=eps_end,
                 exploration_fraction=exploration_fraction,
                 random_episodes=random_episodes,
                 eval_episodes=n_eval_episodes,
                 mean_episodes=mean_episodes,
                 seed=seed,
                 render=render,
                 verbose=verbose)


seeds = [9, 44, 404, 240, 259, 141, 371, 794, 41, 507, 819, 959, 829, 558, 638, 127, 672, 4, 635, 687]
seeds = seeds[:n_runs]

# seeds = [np.random.randint(1000000) for _ in range(n_runs)]

if n_jobs == 1:
    results = [run(mdp,seed) for (mdp,seed) in zip(mdps,seeds)]
elif n_jobs > 1:
    results = Parallel(n_jobs=n_jobs)(delayed(run)(mdp,seed,idx) for (mdp,seed,idx) in zip(mdps,seeds, range(n_runs)))

utils.save_object(results, file_name)