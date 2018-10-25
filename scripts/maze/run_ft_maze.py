import sys
sys.path.append("../")

import numpy as np
from envs.emaze import Maze
from approximators.mlp_torch import MLPQFunction
from operators.mellow_torch import MellowBellmanOperator
from algorithms.nt import learn
import utils
import argparse
from joblib import Parallel, delayed
import datetime
from random import shuffle
from algorithms.dqn import DQN



# Global parameters
render = False
verbose = True
path = os.path.dirname(os.path.realpath(__file__))  # path to this directory

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--kappa", default=100.)
parser.add_argument("--xi", default=0.5)
parser.add_argument("--tau", default=0.0)
parser.add_argument("--batch_size", default=32)
parser.add_argument("--max_iter", default=20000)
parser.add_argument("--buffer_size", default=50000)
parser.add_argument("--random_episodes", default=0)
parser.add_argument("--exploration_fraction", default=0.6)
parser.add_argument("--eps_start", default=1.0)
parser.add_argument("--eps_end", default=0.1)
parser.add_argument("--train_freq", default=1)
parser.add_argument("--eval_freq", default=100)
parser.add_argument("--mean_episodes", default=50)
parser.add_argument("--alpha", default=0.001)
parser.add_argument("--maze", default=6)
parser.add_argument("--n_jobs", default=1)
parser.add_argument("--n_runs", default=1)
parser.add_argument("--l1", default=32)
parser.add_argument("--l2", default=32)
parser.add_argument("--file_name", default="nt_{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
parser.add_argument("--source_file", default="mazes10x10_ns20_32x32nn_dqn_uniqueInit")
parser.add_argument("--dqn", default=False)
parser.add_argument("--mazes_file", default= path + "/mazes10x10")

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
maze = int(args.maze)
l1 = int(args.l1)
l2 = int(args.l2)
n_jobs = int(args.n_jobs)
n_runs = int(args.n_runs)
file_name = str(args.file_name)
source_file = str(args.source_file)
dqn = bool(args.dqn)
mazes_file = args.mazes_file
# Generate tasks

mazes = utils.load_object(mazes_file)

mdps = [Maze(size=maze[0], wall_dim=maze[1], goal_pos=maze[2], start_pos=maze[3], walls=maze[4]) \
            for maze in mazes]

envs = list()
sources = list()
weights = utils.load_object(source_file)
if maze == -1:
    for i in range(min(n_runs, len(mdps))):
        envs.append(mdps[i % len(mdps)])
        sources.append([w for w in weights if not np.array_equal(w[0][-1], envs[-1].walls) and not np.array_equal(w[0][-2], envs[-1].goal)])
else:
    envs = [mdps[maze] for i in range(n_runs)]
    sources = [w for w in weights if
               not np.array_equal(w[0][-1], envs[-1].walls) and not np.array_equal(w[0][-2], envs[-1])]
    #sources = [sources for i in range(n_runs)]


state_dim = mdps[0].state_dim
action_dim = 1
n_actions = mdps[0].action_space.n

# Create Q Function
layers = [l1]
if l2 > 0:
    layers.append(l2)
Q = MLPQFunction(state_dim, n_actions, layers=layers)

if not dqn:
    Q = MLPQFunction(state_dim, n_actions, layers=layers)
    # Create BellmanOperator
    operator = MellowBellmanOperator(kappa, tau, xi, mdps[0].gamma, state_dim, action_dim)
else:
    Q, operator = DQN(state_dim, action_dim, n_actions, mdps[0].gamma, layers=layers)

ws = np.array([w[1] for w in sources])
np.random.shuffle(ws)

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
                 eval_states=None,
                 mean_episodes=mean_episodes,
                 # preprocess=preprocess,
                 seed=seed,
                 render=render,
                 verbose=verbose)


if n_jobs == 1:
    results = [run(mdp) for mdp in mdps]
elif n_jobs > 1:
    seeds = [np.random.randint(1000000) for _ in range(n_runs)]
    results = Parallel(n_jobs=n_jobs)(delayed(run)(mdp,seed, idx) for (mdp,seed, idx) in zip(envs,seeds,range(n_runs)))

utils.save_object(results, file_name)


