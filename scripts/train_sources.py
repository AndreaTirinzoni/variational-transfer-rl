import numpy as np
from features.agrbf import AGaussianRBF
from envs.walled_gridworld import WalledGridworld
from envs.marcellos_gridworld import MarcellosGridworld
from VariationalTransfer.LinearQRegressor import  LinearQRegressor
from scripts.run_gaussian_var_transfer import linearFQI
import utils
from joblib import Parallel, delayed


def train_gridworld(filename, gw_size=5, n_actions=4, n_basis=6, n_sources=5, n_iter=50, epsilon=0.2):
    state_dim = 2
    action_dim = 1
    K = n_basis**2 * n_actions
    x = np.linspace(0, gw_size, n_basis)
    y = np.linspace(0, gw_size, n_basis)
    a = np.linspace(0, n_actions - 1, n_actions)
    mean_x, mean_y, mean_a = np.meshgrid(x, y, a)
    mean = np.hstack((mean_x.reshape(K, 1), mean_y.reshape(K, 1), mean_a.reshape(K, 1)))
    assert mean.shape == (K, 3)

    state_var = (gw_size / (n_basis - 1) / 3) ** 2
    action_var = 0.1 ** 2
    covar = np.eye(state_dim + action_dim)
    covar[0:state_dim, 0:state_dim] *= state_var
    covar[-1, -1] *= action_var
    assert covar.shape == (3, 3)
    covar = np.tile(covar, (K, 1))
    assert covar.shape == (3 * K, 3)

    features = AGaussianRBF(mean, covar, K=K, dims=state_dim + action_dim)
    worlds = list()
    q_functions = list()
    for i in range(n_sources):
        x = np.random.ranf(1)[0] * (gw_size - 1) + 0.5
        worlds.append(WalledGridworld(size=np.array((gw_size, gw_size)), door_x=x))
        q_functions.append(LinearQRegressor(features, np.arange(n_actions), state_dim, action_dim))

    Parallel(n_jobs=20)(delayed(linearFQI)(worlds[i], q_functions[i], epsilon=epsilon, \
                                           n_iter=n_iter, render=False, verbose=False) for _ in range(n_sources))

    weights = np.array([q._w for q in q_functions])
    prior_mean = np.average(weights, axis=0)
    prior_variance = np.average((weights - prior_mean) * (weights - prior_mean), axis=0)
    utils.save_object((prior_mean, prior_variance), filename)

    return prior_mean, prior_variance

def train_marcellos_gw(filename, gw_size=5, n_actions=4, n_basis=6, n_sources=5, n_iter=50, epsilon=0.2):
    state_dim = 2
    action_dim = 1
    K = n_basis ** 2 * n_actions
    x = np.linspace(0, gw_size, n_basis)
    y = np.linspace(0, gw_size, n_basis)
    a = np.linspace(0, n_actions - 1, n_actions)
    mean_x, mean_y, mean_a = np.meshgrid(x, y, a)
    mean = np.hstack((mean_x.reshape(K, 1), mean_y.reshape(K, 1), mean_a.reshape(K, 1)))
    assert mean.shape == (K, 3)

    state_var = (gw_size / (n_basis - 1) / 3) ** 2
    action_var = 0.1 ** 2
    covar = np.eye(state_dim + action_dim)
    covar[0:state_dim, 0:state_dim] *= state_var
    covar[-1, -1] *= action_var
    assert covar.shape == (3, 3)
    covar = np.tile(covar, (K, 1))
    assert covar.shape == (3 * K, 3)

    features = AGaussianRBF(mean, covar, K=K, dims=state_dim + action_dim)
    worlds = list()
    q_functions = list()
    for i in range(n_sources):
        x = np.random.ranf(1)[0] * (gw_size - 1) + 0.5
        y = np.random.ranf(1)[0] * (gw_size - 1) + 0.5
        worlds.append(MarcellosGridworld(size=np.array((gw_size, gw_size)), door_x=(x,y)))
        q_functions.append(LinearQRegressor(features, np.arange(n_actions), state_dim, action_dim))


    Parallel(n_jobs=20)(delayed(linearFQI)(worlds[i], q_functions[i], epsilon=epsilon, \
                                           n_iter=n_iter, render=False, verbose=False) for _ in range(n_sources))

    weights = np.array([q._w for q in q_functions])
    prior_mean = np.average(weights, axis=0)
    prior_variance = np.average((weights - prior_mean) * (weights - prior_mean), axis=0)
    utils.save_object((prior_mean, prior_variance), filename)

    return prior_mean, prior_variance

if __name__ == "__main__":

    train_gridworld("source_wgw5x5_20", n_sources=30, n_iter=50)
    train_marcellos_gw("source_mgw5x5_20", n_sources=30, n_iter=100)