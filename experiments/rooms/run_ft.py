import sys
import os
path = os.path.dirname(os.path.realpath(__file__))  # path to this directory
sys.path.append(os.path.abspath(path + "/../.."))

import numpy as np
from envs.two_room_gw import TwoRoomGridworld
from envs.three_room_gw import ThreeRoomGridworld
from features.agrbf import build_features_gw_state
from approximators.mlp_torch import MLPQFunction
from operators.mellow_torch import MellowBellmanOperator
from algorithms.nt import learn
from misc import utils
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
parser.add_argument("--max_iter", default=20000)
parser.add_argument("--buffer_size", default=50000)
parser.add_argument("--random_episodes", default=0)
parser.add_argument("--exploration_fraction", default=0.2)
parser.add_argument("--eps_start", default=1.0)
parser.add_argument("--eps_end", default=0.02)
parser.add_argument("--train_freq", default=1)
parser.add_argument("--eval_freq", default=50)
parser.add_argument("--mean_episodes", default=50)
parser.add_argument("--alpha", default=0.001)
parser.add_argument("--env", default="three-room-gw")
parser.add_argument("--gw_size", default=10)
# Door at -1 means random positions over all runs
parser.add_argument("--door", default=-1)
parser.add_argument("--door2", default=-1)
parser.add_argument("--n_basis", default=11)
parser.add_argument("--n_jobs", default=1)
parser.add_argument("--n_runs", default=20)
parser.add_argument("--file_name", default="nt_{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
parser.add_argument("--source_file", default=path + "/sources")

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
door2 = float(args.door2)
n_basis = int(args.n_basis)
n_jobs = int(args.n_jobs)
n_runs = int(args.n_runs)
file_name = str(args.file_name)
source_file = str(args.source_file)

# Load sources
weights = utils.load_object(source_file)
ws = np.array([w[1] for w in weights])
np.random.shuffle(ws)
params = np.array([np.array(w[0][1]) for w in weights])

n_runs = min((n_runs, len(weights)))

# Generate tasks
doors = [np.random.uniform(0.5, gw_size - 0.5) if door < 0 else door for _ in range(n_runs)]
doors2 = [np.random.uniform(0.5, gw_size - 0.5) if door2 < 0 else door2 for _ in range(n_runs)]
if env == "two-room-gw":
    mdps = [TwoRoomGridworld(np.array([gw_size, gw_size]), door_x=d) for d in doors]
elif env == "three-room-gw":
    mdps = [ThreeRoomGridworld(np.array([gw_size, gw_size]), door_x=(d1,d2)) for (d1,d2) in zip(doors,doors2)]
eval_states = [np.array([0., 0.]) for _ in range(10)]

state_dim = mdps[0].state_dim
action_dim = 1
n_actions = mdps[0].action_space.n
K = n_basis ** 2

# Create BellmanOperator
operator = MellowBellmanOperator(kappa, tau, xi, mdps[0].gamma, K, action_dim)
# Create Q Function
Q = MLPQFunction(K, n_actions, layers=None)
# Create RBFs
rbf = build_features_gw_state(gw_size, n_basis, state_dim)




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
                 eval_states=eval_states,
                 mean_episodes=mean_episodes,
                 preprocess=rbf,
                 seed=seed,
                 render=render,
                 verbose=verbose)


seeds = [np.random.randint(1000000) for _ in range(n_runs)]

if n_jobs == 1:
    results = [run(mdp,seed) for (mdp,seed) in zip(mdps,seeds)]
elif n_jobs > 1:
    results = Parallel(n_jobs=n_jobs)(delayed(run)(mdp,seed,idx) for (mdp,seed,idx) in zip(mdps,seeds,range(n_runs)))

utils.save_object(results, file_name)
