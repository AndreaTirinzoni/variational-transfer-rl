import numpy as np
from features.agrbf import AGaussianRBF
from envs.walled_gridworld import WalledGridworld
from VariationalTransfer.LinearQRegressor import LinearQRegressor
from algorithms.e_greedy_policy import eGreedyPolicy
import utils
import argparse
from joblib import Parallel, delayed

import sys
sys.path.append("../")

# Global parameters
kappa = 100.
gamma = 0.99
xi = 1.0
tau = 0.0
batch_size = 1
gradient_batch = 1000
epsilon = 1
max_iter = 300
n_fit = 1
gw_size = 5
n_actions = 4
state_dim = 2
action_dim = 1
n_basis = 6
render = False
verbose = True
n_jobs = 1
n_tasks = 1

# Adam params
m_t = 0
v_t = 0
t = 0
eps = 1e-8
alpha = 0.01
beta_1 = 0.9
beta_2 = 0.999


def gradient(Q, data):

    r = Q.get_statedim() + Q.get_actiondim()
    s_prime = r + 1
    br = bellman_residual(Q, data)
    assert br.shape == (data.shape[0],)
    mm_gradient = gradient_mm(Q, data[:, s_prime:s_prime+Q.get_statedim()], data[:, -1])
    assert mm_gradient.shape == (data.shape[0], K)
    q_gradient = Q.compute_gradient(data[:, 0:r])
    assert q_gradient.shape == (data.shape[0], K)
    b_grad = xi * gamma * mm_gradient - q_gradient
    assert b_grad.shape == (data.shape[0], K)
    bellman_grad = 2 * np.sum(br[:, np.newaxis] * b_grad * softmax(br ** 2)[:, np.newaxis], axis=0)
    assert bellman_grad.shape == (K,)

    return bellman_grad


def gradient_mm(Q, states, done):

    q_values = Q.compute_all_actions(states, done)
    assert q_values.shape == (states.shape[0],n_actions)
    q_gradient = Q.compute_gradient_all_actions(states) * (1 - done)[:, np.newaxis, np.newaxis]
    assert q_gradient.shape == (states.shape[0],n_actions,K)
    qs = mm_exp(q_values, np.max(q_values, axis=1))
    assert qs.shape == (states.shape[0],n_actions)
    qs_sum = np.sum(qs, axis=1)
    assert qs_sum.shape == (states.shape[0],)
    grad = np.sum(qs[:, :, np.newaxis] * q_gradient, axis=1) / qs_sum[:,np.newaxis]
    assert grad.shape == (states.shape[0], K)

    return grad


def bellman_residual(Q, data):

    r = Q.get_statedim() + Q.get_actiondim()
    s_prime = r + 1
    feats_s_prime = Q.compute_gradient_all_actions(data[:, s_prime:s_prime + Q.get_statedim()])
    assert feats_s_prime.shape == (data.shape[0],n_actions,K)
    feats = Q.compute_gradient(data[:, 0:r])
    assert feats.shape == (data.shape[0], K)
    Qs = np.dot(feats, Q._w)
    assert Qs.shape == (data.shape[0],)
    Qs_prime = np.dot(feats_s_prime, Q._w)
    assert Qs_prime.shape == (data.shape[0],n_actions)
    mmQs = mellow_max(Qs_prime)
    assert mmQs.shape == (data.shape[0],)

    return data[:, r] + gamma * mmQs * (1 - data[:, -1]) - Qs


def mellow_max(X):
    mx = np.max(X, axis=1)
    assert mx.shape == (X.shape[0],)
    qs = np.sum(mm_exp(X, mx), axis=1)
    assert qs.shape == (X.shape[0],)
    return np.log(qs/X.shape[1]) / kappa + mx


def mm_exp(X, c=0):
    return np.squeeze(np.exp(kappa * (X - c[:, np.newaxis])))


def softmax(X):
    mx = np.max(X)
    num = np.exp(tau * (X - mx))
    return num / np.sum(num)


def adam(w, grad):
    global t
    global m_t
    global v_t

    t += 1
    m_t = beta_1 * m_t + (1 - beta_1) * grad
    v_t = beta_2 * v_t + (1 - beta_2) * grad ** 2
    m_t_hat = m_t / (1 - beta_1 ** t)
    v_t_hat = v_t / (1 - beta_2 ** t)
    return w - alpha * m_t_hat / (np.sqrt(v_t_hat) + eps)


def run(door_x, seed=None):

    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(0, gw_size, n_basis)
    y = np.linspace(0, gw_size, n_basis)
    a = np.linspace(0, n_actions - 1, n_actions)
    mean_x, mean_y, mean_a = np.meshgrid(x, y, a)
    mean = np.hstack((mean_x.reshape(K, 1), mean_y.reshape(K, 1), mean_a.reshape(K, 1)))
    assert mean.shape == (K, 3)

    state_var = (gw_size / (n_basis - 1) / 3) ** 2
    action_var = 0.01 ** 2
    covar = np.eye(state_dim + action_dim)
    covar[0:state_dim, 0:state_dim] *= state_var
    covar[-1, -1] *= action_var
    assert covar.shape == (3, 3)
    covar = np.tile(covar, (K, 1))
    assert covar.shape == (3 * K, 3)

    # Features
    features = AGaussianRBF(mean, covar, K=K, dims=state_dim + action_dim)

    # Create Target task
    mdp = WalledGridworld(np.array([gw_size, gw_size]), door_x=door_x)
    Q = LinearQRegressor(features, np.arange(n_actions), state_dim, action_dim)

    # Learning
    pi = eGreedyPolicy(Q, Q.actions, epsilon=epsilon)
    pi_g = eGreedyPolicy(Q, Q.actions, epsilon=0)
    pi_u = eGreedyPolicy(Q, Q.actions, epsilon=1)

    samples = utils.generate_episodes(mdp, pi_u, n_episodes=1, render=False)

    # Results
    iterations = []
    n_samples = []
    rewards = []
    l_2 = []
    l_inf = []
    sft = []

    for i in range(max_iter):
        new_samples = utils.generate_episodes(mdp, pi, n_episodes=batch_size, render=False)
        samples = np.vstack((samples, new_samples))
        for _ in range(n_fit):
            np.random.shuffle(samples)
            grad = gradient(Q, samples[:gradient_batch, 1:])
            Q._w = adam(Q._w, grad)
        #utils.plot_Q(Q)

        rew = utils.evaluate_policy(mdp, pi_g, render=render, initial_states=[np.array([0., 0.]) for _ in range(10)])
        br = bellman_residual(Q, samples[:, 1:]) ** 2
        l_2_err = np.average(br)
        l_inf_err = np.max(br)
        sft_err = np.sum(softmax(br) * br)

        iterations.append(i)
        n_samples.append(samples.shape[0])
        rewards.append(rew)
        l_2.append(l_2_err)
        l_inf.append(l_inf_err)
        sft.append(sft_err)

        if verbose:
            print("===============================================")
            print("Door X: " + str(door_x))
            print("Iteration " + str(i))
            print("Reward: " + str(rew))
            print("L2 Error: " + str(l_2_err))
            print("L_inf Error: " + str(l_inf_err))
            print("Softmax Error: " + str(sft_err))
            print("===============================================")

    run_info = [iterations, n_samples, rewards, l_2, l_inf, sft]
    weights = np.array(Q._w)

    return [door_x, weights, run_info]


parser = argparse.ArgumentParser()
parser.add_argument("--kappa", default=kappa)
parser.add_argument("--xi", default=xi)
parser.add_argument("--tau", default=tau)
parser.add_argument("--batch_size", default=batch_size)
parser.add_argument("--gradient_batch", default=gradient_batch)
parser.add_argument("--max_iter", default=max_iter)
parser.add_argument("--n_fit", default=n_fit)
parser.add_argument("--alpha", default=alpha)
parser.add_argument("--gw_size", default=gw_size)
parser.add_argument("--n_basis", default=n_basis)
parser.add_argument("--n_jobs", default=n_jobs)
parser.add_argument("--n_tasks", default=n_tasks)
parser.add_argument("--file_name", default="mm")

args = parser.parse_args()
kappa = float(args.kappa)
xi = float(args.xi)
tau = float(args.tau)
batch_size = int(args.batch_size)
gradient_batch = int(args.gradient_batch)
max_iter = int(args.max_iter)
n_fit = int(args.n_fit)
alpha = float(args.alpha)
gw_size = int(args.gw_size)
n_basis = int(args.n_basis)
K = n_basis ** 2 * n_actions
n_jobs = int(args.n_jobs)
n_tasks = int(args.n_tasks)
file_name = str(args.file_name)

doors = [np.random.uniform(0.5, gw_size - 0.5) for _ in range(n_tasks)]

if n_jobs == 1:
    results = [run(door) for door in doors]
elif n_jobs > 1:
    seeds = [np.random.randint(1000000) for _ in range(n_tasks)]
    results = Parallel(n_jobs=n_jobs)(delayed(run)(door,seed) for (door,seed) in zip(doors,seeds))

utils.save_object(results, file_name)