import sys
sys.path.append("../")

import numpy as np
from envs.cartpole import CartPoleEnv
from approximators.mlp import MLPQFunction
from operators.mellow import MellowBellmanOperator
from algorithms.gvt import learn
import utils
import argparse
from joblib import Parallel, delayed
import datetime


# Global parameters
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
parser.add_argument("--n_source", default=50)
parser.add_argument("--sigma_reg", default=0.0001)
parser.add_argument("--cholesky_clip", default=0.0001)
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
cart_mass = float(args.cart_mass)
pole_mass = float(args.pole_mass)
pole_length = float(args.pole_length)
n_jobs = int(args.n_jobs)
n_runs = int(args.n_runs)
file_name = str(args.file_name)
source_file = str(args.source_file)

# Seed to get reproducible results
np.random.seed(485)

# Generate tasks
mc = [np.random.uniform(0.5, 1.5) if cart_mass < 0 else cart_mass for _ in range(n_runs)]
mp = [np.random.uniform(0.1, 0.2) if pole_mass < 0 else pole_mass for _ in range(n_runs)]
l = [np.random.uniform(0.2, 0.8) if pole_length < 0 else pole_length for _ in range(n_runs)]
mdps = [CartPoleEnv(a,b,c) for a,b,c in zip(mc,mp,l)]
n_eval_episodes = 5

state_dim = mdps[0].state_dim
action_dim = 1
n_actions = mdps[0].action_space.n

# Create BellmanOperator
operator = MellowBellmanOperator(kappa, tau, xi, mdps[0].gamma, state_dim, action_dim)
# Create Q Function
layers = [l1]
if l2 > 0:
    layers.append(l2)
Q = MLPQFunction(state_dim, n_actions, layers=layers)


def run(mdp, seed=None):
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
                 sigma_reg=sigma_reg,
                 cholesky_clip=cholesky_clip,
                 time_coherent=time_coherent,
                 n_source=n_source,
                 source_file=source_file,
                 seed=seed,
                 render=render,
                 verbose=verbose)


seeds = [9, 44, 404, 240, 259, 141, 371, 794, 41, 507, 819, 959, 829, 558, 638, 127, 672, 4, 635, 687]
seeds = seeds[:n_runs]
if n_jobs == 1:
    results = [run(mdp,seed) for (mdp,seed) in zip(mdps,seeds)]
elif n_jobs > 1:
    results = Parallel(n_jobs=n_jobs)(delayed(run)(mdp,seed) for (mdp,seed) in zip(mdps,seeds))

utils.save_object(results, file_name)


