import numpy as np
from features.agrbf import AGaussianRBF
from envs.walled_gridworld import WalledGridworld
from VariationalTransfer.LinearQRegressor import LinearQRegressor
from algorithms.e_greedy_policy import eGreedyPolicy
import utils

# Global parameters
kappa = 100.
gamma = 0.99
eta = 0.1
batch_size = 1
epsilon = 0.2
max_iter = 100
n_fit = 1
gw_size = 5
n_actions = 4
state_dim = 2
action_dim = 1
n_basis = 6
K = n_basis ** 2 * n_actions
render = False
verbose = True

def gradient(Q, data):

    r = Q.get_statedim() + Q.get_actiondim()
    s_prime = r + 1
    br = bellman_residual(Q, data)
    assert br.shape == (data.shape[0],)
    mm_gradient = gradient_mm(Q, data[:, s_prime:s_prime+Q.get_statedim()])
    assert mm_gradient.shape == (data.shape[0], 144)
    q_gradient = Q.compute_gradient(data[:, 0:r])
    assert q_gradient.shape == (data.shape[0], 144)
    b_grad = gamma * mm_gradient - q_gradient
    assert b_grad.shape == (data.shape[0], 144)
    bellman_grad = 2 * np.average(br[:, np.newaxis] * b_grad, axis=0)
    assert bellman_grad.shape == (144,)

    return bellman_grad

def gradient_mm(Q, states):
    q_values = Q.compute_all_actions(states)
    assert q_values.shape == (states.shape[0],4)
    q_gradient = Q.compute_gradient_all_actions(states)
    assert q_gradient.shape == (states.shape[0],4,144)
    qs = mm_exp(q_values, np.max(q_values, axis=1))
    assert qs.shape == (states.shape[0],4)
    qs_sum = np.sum(qs, axis=1)
    assert qs_sum.shape == (states.shape[0],)
    grad = np.sum(qs[:, :, np.newaxis] * q_gradient, axis=1) / qs_sum[:,np.newaxis]
    assert grad.shape == (states.shape[0], 144)

    return grad

def bellman_residual(Q, data):

    r = Q.get_statedim() + Q.get_actiondim()
    s_prime = r + 1
    feats_s_prime = Q.compute_gradient_all_actions(data[:, s_prime:s_prime + Q.get_statedim()])
    assert feats_s_prime.shape == (data.shape[0],4,144)
    feats = Q.compute_gradient(data[:, 0:r])
    assert feats.shape == (data.shape[0], 144)
    Qs = np.dot(feats, Q._w)
    assert Qs.shape == (data.shape[0],)
    Qs_prime = np.dot(feats_s_prime, Q._w)
    assert Qs_prime.shape == (data.shape[0],4)
    mmQs = mellow_max(Qs_prime)
    assert mmQs.shape == (data.shape[0],)

    return data[:, r] + gamma * mmQs - Qs


def mellow_max(X):
    mx = np.max(X, axis=1)
    assert mx.shape == (X.shape[0],)
    qs = np.sum(mm_exp(X, mx), axis=1)
    assert qs.shape == (X.shape[0],)
    return np.log(qs/X.shape[1]) / kappa + mx


def mm_exp(X, c=0):
    return np.squeeze(np.exp(kappa * (X - c[:, np.newaxis])))



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

# Features
features = AGaussianRBF(mean, covar, K=K, dims=state_dim + action_dim)

# Create Target task
mdp = WalledGridworld(np.array([gw_size, gw_size]), door_x=2.5)
Q = LinearQRegressor(features, np.arange(n_actions), state_dim, action_dim)

# Learning
pi = eGreedyPolicy(Q, Q.actions, epsilon=epsilon)
pi_g = eGreedyPolicy(Q, Q.actions, epsilon=0)
pi_u = eGreedyPolicy(Q, Q.actions, epsilon=1)

samples = utils.generate_episodes(mdp, pi_u, n_episodes=1, render=False)

for i in range(max_iter):
    new_samples = utils.generate_episodes(mdp, pi, n_episodes=batch_size, render=False)
    samples = np.vstack((samples, new_samples))
    for _ in range(n_fit):
        grad = gradient(Q, samples[:, 1:])
        Q._w = Q._w - eta * grad
    utils.plot_Q(Q)
    rew = utils.evaluate_policy(mdp, pi_g, render=render, initial_states=np.array([0.,0.]))

    if verbose:
        print("===============================================")
        print("Iteration " + str(i))
        print("Reward: " + str(rew))
        print("Error: " + str(np.average(bellman_residual(Q, samples[:, 1:]) ** 2)))
        print("===============================================")

