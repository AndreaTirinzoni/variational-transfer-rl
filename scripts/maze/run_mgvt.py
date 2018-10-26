import sys
import os
path = os.path.dirname(os.path.realpath(__file__))  # path to this directory
sys.path.append(os.path.abspath(path + "/../.."))

from envs.emaze import Maze
from approximators.mlp_torch import MLPQFunction
from operators.mellow_torch import MellowBellmanOperator
from algorithms.mgvt_torch import learn
from misc import utils
import argparse
from joblib import Parallel, delayed
import datetime
import numpy as np


# Global parameters
render = False
verbose = True
path = os.path.dirname(os.path.realpath(__file__))  # path to this directory

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--kappa", default=100.)
parser.add_argument("--xi", default=0.5)
parser.add_argument("--tau", default=0.0)
parser.add_argument("--batch_size", default=50)
parser.add_argument("--max_iter", default=20000)
parser.add_argument("--buffer_size", default=10000)
parser.add_argument("--random_episodes", default=0)
parser.add_argument("--train_freq", default=1)
parser.add_argument("--eval_freq", default=100)
parser.add_argument("--mean_episodes", default=25)
parser.add_argument("--alpha_adam", default=0.001)
parser.add_argument("--alpha_sgd", default=0.0001)
parser.add_argument("--lambda_", default=0.00001)
parser.add_argument("--time_coherent", default=False)
parser.add_argument("--n_weights", default=10)
parser.add_argument("--n_source", default=10)
parser.add_argument("--cholesky_clip", default=0.0001)
# Door at -1 means random positions over all runs
parser.add_argument("--maze", default=-1)
parser.add_argument("--l1", default=32)
parser.add_argument("--l2", default=32)
parser.add_argument("--mazes_file", default=path + "/mazes10x10")
parser.add_argument("--n_jobs", default=1)
parser.add_argument("--n_runs", default=20)
parser.add_argument("--file_name", default="mgvt_{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
parser.add_argument("--source_file", default=path + "/sources")
parser.add_argument("--eta", default=1e-6)  # learning rate for
parser.add_argument("--eps", default=0.001)  # precision for the initial posterior approximation and upperbound tighting
parser.add_argument("--bandwidth", default=.00001)  # Bandwidth for the Kernel Estimator
parser.add_argument("--post_components", default=1)  # number of components of the posterior family
parser.add_argument("--max_iter_ukl", default=60)
parser.add_argument("--fixed_seed", default=-1)

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
alpha_adam = float(args.alpha_adam)
alpha_sgd = float(args.alpha_sgd)
lambda_ = float(args.lambda_)
time_coherent = bool(int(args.time_coherent))
n_weights = int(args.n_weights)
n_source = int(args.n_source)
cholesky_clip = float(args.cholesky_clip)
l1 = int(args.l1)
l2 = int(args.l2)
n_jobs = int(args.n_jobs)
n_runs = int(args.n_runs)
file_name = str(args.file_name)
source_file = str(args.source_file)
eps = float(args.eps)
eta = float(args.eta)
post_components = int(args.post_components)
bandwidth = float(args.bandwidth)
max_iter_ukl = int(args.max_iter_ukl)
fixed_seed = int(args.fixed_seed)

n_eval_episodes = 10
mazes_file = args.mazes_file
maze = int(args.maze)

# Seed to get reproducible results
np.random.seed(485)

# Create Environments
mazes = utils.load_object(mazes_file)
weights = utils.load_object(source_file)

mdps = [Maze(size=maze[0], wall_dim=maze[1], goal_pos=maze[2], start_pos=maze[3], walls=maze[4]) \
            for maze in mazes]

envs = list()
sources = list()

if maze == -1:
    for i in range(min(n_runs, len(mdps))):
        envs.append(mdps[i % len(mdps)])
        sources.append([w for w in weights if not np.array_equal(w[0][-1], envs[-1].walls) and not np.array_equal(w[0][-2], envs[-1].goal)])
else:
    envs = [mdps[maze] for i in range(n_runs)]
    sources = [w for w in weights if
               not np.array_equal(w[0][-1], envs[-1].walls) and not np.array_equal(w[0][-2], envs[-1])]
    sources = [sources for i in range(n_runs)]

state_dim = mdps[0].state_dim
action_dim = 1
n_actions = mdps[0].action_space.n

# Create Q Function
layers = [l1]
if l2 > 0:
    layers.append(l2)

Q = MLPQFunction(state_dim, n_actions, layers=layers)
# Create BellmanOperator
operator = MellowBellmanOperator(kappa, tau, xi, mdps[0].gamma, state_dim, action_dim)

def run(mdp, seed=None, source=None):
    return learn(mdp,
                 Q,
                 operator,
                 max_iter=max_iter,
                 buffer_size=buffer_size,
                 batch_size=batch_size,
                 alpha_adam=alpha_adam,
                 alpha_sgd=alpha_sgd,
                 lambda_=lambda_,
                 n_weights=n_weights,
                 train_freq=train_freq,
                 eval_freq=eval_freq,
                 random_episodes=random_episodes,
                 eval_episodes=n_eval_episodes,
                 mean_episodes=mean_episodes,
                 cholesky_clip=cholesky_clip,
                 bandwidth=bandwidth,
                 post_components=post_components,
                 max_iter_ukl=max_iter_ukl,
                 eps=eps,
                 eta=eta,
                 time_coherent=time_coherent,
                 n_source=n_source,
                 source_file=source_file,
                 seed=seed,
                 render=render,
                 verbose=verbose,
                 ukl_tight_freq=1,
                 sources=source)


if fixed_seed < 0:
    seeds = [9, 44, 404, 240, 259, 141, 371, 794, 41, 507, 819, 959, 829, 558, 638, 127, 672, 4, 635, 687]
    seeds = seeds[:n_runs]
else:
    seeds = [fixed_seed for _ in range(n_runs)]

if n_jobs == 1:
    results = [run(mdp,seed) for (mdp,seed,source) in zip(envs,seeds,sources)]
elif n_jobs > 1:
    results = Parallel(n_jobs=n_jobs)(delayed(run)(mdp,seed,source) for (mdp,seed,source) in zip(envs,seeds,sources))

file_name = file_name + "_seed" + str(fixed_seed) if fixed_seed > 0 else file_name
utils.save_object(results, file_name)
