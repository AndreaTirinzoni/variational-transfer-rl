import numpy as np
from features.agrbf import AGaussianRBF
from envs.walled_gridworld import WalledGridworld
from envs.marcellos_gridworld import MarcellosGridworld
from VariationalTransfer.LinearQRegressor import  LinearQRegressor
from VariationalTransfer.Distributions import AnisotropicNormalPosterior
from VariationalTransfer.BellmanOperator import LinearQBellmanOperator
from VariationalTransfer.VarTransfer import VarTransferGaussian
import utils

gw_size = 5
n_actions = 4
state_dim = 2
action_dim = 1
n_basis = 6
K = n_basis ** 2 * n_actions

x = np.linspace(0, gw_size, n_basis)
y = np.linspace(0, gw_size, n_basis)
a = np.linspace(0, n_actions - 1, n_actions)
mean_x, mean_y, mean_a = np.meshgrid(x, y, a)
mean = np.hstack((mean_x.reshape(K, 1), mean_y.reshape(K, 1), mean_a.reshape(K, 1)))
assert mean.shape == (K,3)

state_var = (gw_size / (n_basis - 1) / 3) ** 2
action_var = 0.1 ** 2
covar = np.eye(state_dim + action_dim)
covar[0:state_dim, 0:state_dim] *= state_var
covar[-1, -1] *= action_var
assert covar.shape == (3,3)
covar = np.tile(covar, (K, 1))
assert covar.shape == (3*K,3)

# features
features = AGaussianRBF(mean, covar, K=K, dims=state_dim + action_dim)

prior_mean = np.zeros(K)
prior_variance = np.ones(K)*0.1

prior_mean, prior_variance = utils.load_object("source_mgw5x5_20")

# Create Target task
mdp = MarcellosGridworld(np.array([gw_size, gw_size]), door_x=(4.5, 0.5))
q = LinearQRegressor(features, np.arange(n_actions), state_dim, action_dim)
prior = AnisotropicNormalPosterior(prior_mean, prior_variance)
x=prior.sample(nsamples=5)
bellman = LinearQBellmanOperator(q, gamma=mdp.gamma)
# bellman = LinearQMellowBellman(q, gamma=mdp.gamma)
var = VarTransferGaussian(mdp, bellman, prior, learning_rate=1e-4, likelihood_weight=100)

r = list()
for _ in range(10):
    rew = var.solve_task(max_iter=100, n_fit=1, batch_size=1, verbose=True, render=False)
    r.append(rew)
    var.reset()