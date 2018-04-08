import numpy as np
from features.agrbf import AGaussianRBF
from VariationalTransfer.LinearQRegressor import LinearQRegressor
from envs.walled_gridworld import WalledGridworld
from algorithms.e_greedy_policy import eGreedyPolicy
import utils

# Global parameters
kappa = 100.
gamma = 0.99
xi = 1.0
tau = 0.0
eta = 0.1
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
K = n_basis ** 2 * n_actions
render = False
verbose = True
n_weights = 100
lambda_ = 1e-3

# Adam params
m_t = 0
v_t = 0
t = 0
eps = 1e-8
alpha = 0.001
beta_1 = 0.9
beta_2 = 0.999


def unpack(params):
    """Unpacks a parameter vector into mu and L"""
    mu = params[:K]
    L = params[K:].reshape(K,K)
    return mu, L


def pack(mu, L):
    """Packs mu and L into a parameter vector"""
    return np.concatenate((mu, L.flatten()))


def clip(params):
    """Makes sure the Cholensky factor L is well-defined"""
    mu, L = unpack(params)
    mask = np.logical_and(L < 0.01, np.eye(K, dtype=bool))
    L[mask] = 0.01
    L[np.triu_indices(K, 1)] = 0
    return np.concatenate((mu, L.flatten()))


def sample_posterior(params):
    """Samples a Q function from the posterior distribution"""
    mu, L = unpack(params)
    return np.dot(L, np.random.randn(K,)) + mu


def objective(params):
    """Computes the negative ELBO"""
    mu, L = unpack(params)
    assert mu.shape == (K,) and L.shape == (K,K)
    Sigma = np.dot(L, L.T)
    assert Sigma.shape == (K,K) and np.all(np.linalg.eigvals(Sigma) > 0)
    weights = np.dot(np.random.randn(n_weights, K), L.T) + mu
    assert weights.shape == (n_weights,K)
    likelihood = expected_bellman_error(weights)
    assert likelihood >= 0
    kl = KL(mu, Sigma, mu_bar, Sigma_bar_inv)
    assert kl >= 0
    return likelihood + lambda_ * kl / samples.shape[0]


def expected_bellman_error(weights):
    """Approximates the expected Bellman error with a set of weights"""
    br = bellman_residual(weights) ** 2
    assert br.shape == (samples.shape[0],weights.shape[0])
    errors = np.average(br, axis=0, weights=softmax(br, axis=0))
    assert errors.shape == (weights.shape[0],)
    return np.average(errors)


def bellman_residual(weights):
    """Computes the Bellman residuals of a set of samples given a set of weights"""
    _, _, _, r, s_prime, absorbing, sa = utils.split_data(samples, state_dim, action_dim)
    feats_s_prime = Q.compute_gradient_all_actions(s_prime)
    assert feats_s_prime.shape == (samples.shape[0], n_actions, K)
    feats = Q.compute_gradient(sa)
    assert feats.shape == (samples.shape[0], K)
    Q_values = np.dot(feats, weights.T)
    assert Q_values.shape == (samples.shape[0], weights.shape[0])
    Q_values_prime = np.dot(feats_s_prime, weights.T)
    assert Q_values_prime.shape == (samples.shape[0], n_actions, weights.shape[0])
    mm = mellow_max(Q_values_prime, axis=1)
    assert mm.shape == (samples.shape[0], weights.shape[0])

    return r[:, np.newaxis] + gamma * mm * (1 - absorbing[:, np.newaxis]) - Q_values


def KL(mu1, Sigma1, mu2, Sigma2_inv):
    """Computes the KL divergence between two multivariate Gaussian distributions"""
    mu_diff = mu1 - mu2
    return (np.log(1 / (np.linalg.det(Sigma1) * np.linalg.det(Sigma2_inv))) +
            np.trace(np.dot(Sigma2_inv,Sigma1)) +
            np.dot(np.dot(mu_diff.T,Sigma2_inv),mu_diff) - mu1.shape[0]) / 2


def gradient_KL(mu, L, mu_bar, Sigma_bar_inv):
    """Computes the gradient of the KL between two multivariate Gaussians wrt mean and Cholensky factor of the first"""
    grad_mu = np.dot(Sigma_bar_inv, mu - mu_bar)
    grad_L = np.dot(Sigma_bar_inv, L) - np.linalg.inv(L).T
    return grad_mu, grad_L


def gradient_be(weights):
    """Computes the gradient of the Bellman error for different weights"""
    br = bellman_residual(weights)
    assert br.shape == (samples.shape[0], weights.shape[0])
    mm_grad = gradient_mm(weights)
    assert mm_grad.shape == (samples.shape[0], weights.shape[0], K)
    q_grad = Q.compute_gradient(samples[:, 1:4])
    assert q_grad.shape == (samples.shape[0], K)
    res_grad = xi * gamma * mm_grad - q_grad[:, np.newaxis, :]
    assert res_grad.shape == (samples.shape[0], weights.shape[0], K)
    be_grad = 2 * np.sum(br[:, :, np.newaxis] * res_grad * softmax(br ** 2, axis=0)[:, :, np.newaxis], axis=0)
    assert be_grad.shape == (weights.shape[0], K)

    return be_grad


def gradient_mm(weights):
    """Computes the mellowmax gradient for different weights"""
    _, _, _, _, s_prime, absorbing, _ = utils.split_data(samples, state_dim, action_dim)
    feats_s_prime = Q.compute_gradient_all_actions(s_prime)
    assert feats_s_prime.shape == (samples.shape[0], n_actions, K)
    Q_values_prime = np.dot(feats_s_prime, weights.T) * (1 - absorbing[:, np.newaxis, np.newaxis])
    assert Q_values_prime.shape == (samples.shape[0], n_actions, weights.shape[0])
    feats_s_prime = feats_s_prime * (1 - absorbing)[:, np.newaxis, np.newaxis]
    assert feats_s_prime.shape == (samples.shape[0], n_actions, K)
    sft_Q = softmax(Q_values_prime, temp=kappa, axis=1)
    assert sft_Q.shape == (samples.shape[0], n_actions, weights.shape[0])
    mm_grad = np.squeeze(np.sum(sft_Q[:, :, :, np.newaxis] * feats_s_prime[:, :, np.newaxis, :], axis=1))
    assert mm_grad.shape == (samples.shape[0], weights.shape[0], K)

    return mm_grad


def gradient(params):
    """Computes the objective function gradient"""
    mu, L = unpack(params)
    assert mu.shape == (K,) and L.shape == (K,K)
    Sigma = np.dot(L, L.T)
    assert Sigma.shape == (K,K) and np.all(np.linalg.eigvals(Sigma) > 0)
    vs = np.random.randn(n_weights, K)
    assert vs.shape == (n_weights, K)
    ws = np.dot(vs, L.T) + mu
    assert ws.shape == (n_weights,K)
    be_grad = gradient_be(ws)
    assert be_grad.shape == (ws.shape[0], K)
    ebe_grad_mu = np.average(be_grad, axis=0)
    assert ebe_grad_mu.shape == (K,)
    ebe_grad_L = np.average(be_grad[:, :, np.newaxis] * vs[:, np.newaxis, :], axis=0)
    assert ebe_grad_L.shape == (K,K)
    kl_grad_mu, kl_grad_L = gradient_KL(mu, L, mu_bar, Sigma_bar_inv)
    assert kl_grad_mu.shape == (K,) and kl_grad_L.shape == (K,K)
    grad_mu = ebe_grad_mu + lambda_ * kl_grad_mu / samples.shape[0]
    assert grad_mu.shape == (K,)
    grad_L = ebe_grad_L + lambda_ * kl_grad_L / samples.shape[0]
    assert grad_L.shape == (K,K)

    return pack(grad_mu, grad_L)


def mellow_max(X, axis=-1):
    """Computes the mellow-max of tensor X on the given axis"""
    mx = np.max(X, axis=axis, keepdims=True)
    mm_exp = np.exp(kappa * (X - mx))
    mm_sum = np.sum(mm_exp, axis=axis, keepdims=True)
    return np.squeeze(np.log(mm_sum / X.shape[axis]) / kappa + mx)


def softmax(X, temp=tau, axis=-1):
    """Computes the softmax of tensor X on the given axis"""
    mx = np.max(X, axis=axis, keepdims=True)
    num = np.exp(temp * (X - mx))
    return num / np.sum(num, axis=axis, keepdims=True)


def adam(params, grad):
    global t
    global m_t
    global v_t

    t += 1
    m_t = beta_1 * m_t + (1 - beta_1) * grad
    v_t = beta_2 * v_t + (1 - beta_2) * grad ** 2
    m_t_hat = m_t / (1 - beta_1 ** t)
    v_t_hat = v_t / (1 - beta_2 ** t)
    return params - alpha * m_t_hat / (np.sqrt(v_t_hat) + eps)


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
mdp = WalledGridworld(np.array([gw_size, gw_size]), door_x=0.5)
Q = LinearQRegressor(features, np.arange(n_actions), state_dim, action_dim)

weights = utils.load_object("../scripts/mm_sources_5x5_basis6_rw1_kappa100_xi1_tau0")
ws = np.array([w[1] for w in weights])
mu_bar = np.mean(ws, axis=0)
Sigma_bar = np.cov(ws.T) + np.eye(K) * 0.01
Sigma_bar_inv = np.linalg.inv(Sigma_bar)
params = pack(mu_bar, np.linalg.cholesky(Sigma_bar))

# Learning
pi_u = eGreedyPolicy(Q, Q.actions, epsilon=1)
pi_g = eGreedyPolicy(Q, Q.actions, epsilon=0)

samples = utils.generate_episodes(mdp, pi_u, n_episodes=1, render=False)

# Results
iterations = []
n_samples = []
rewards = []
l_2 = []
l_inf = []
sft = []

for i in range(max_iter):

    s = mdp.reset()
    h = 0
    while h < mdp.horizon:
        Q._w = sample_posterior(params)
        a = np.argmax(Q.compute_all_actions(s))
        s_prime, r, done, _ = mdp.step(a)
        sample = np.concatenate([np.array([t]), s, np.array([a]), np.array([r]), s_prime, np.array([1 if done else 0])])[np.newaxis, :]
        samples = np.concatenate((samples,sample), axis=0)

        for _ in range(n_fit):
            np.random.shuffle(samples)
            g = gradient(params)
            params = clip(adam(params, g))

        s = s_prime
        h += 1
        if done:
            break

    mu, _ = unpack(params)
    Q._w = mu
    utils.plot_Q(Q)
    rew = utils.evaluate_policy(mdp, pi_g, render=render, initial_states=[np.array([0., 0.]) for _ in range(10)])
    br = np.squeeze(bellman_residual(np.concatenate([mu[np.newaxis, :],mu[np.newaxis, :]])) ** 2)
    l_2_err = np.average(br)
    l_inf_err = np.max(br)
    sft_err = np.sum(softmax(br) * br)
    fval = objective(params)

    iterations.append(i)
    n_samples.append(samples.shape[0])
    rewards.append(rew)
    l_2.append(l_2_err)
    l_inf.append(l_inf_err)
    sft.append(sft_err)

    if verbose:
        print("Iteration {} Reward {} Fval {} L2 {} L_inf {} Sft {}".format(i,rew[0],fval,l_2_err,l_inf_err,sft_err))
