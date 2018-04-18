import sys
sys.path.append("../")

import numpy as np
from envs.walled_gridworld import WalledGridworld
from envs.marcellos_gridworld import MarcellosGridworld
from features.agrbf import build_features_gw
from approximators.linear import LinearQFunction
from operators.mellow import MellowBellmanOperator
from policies import EpsilonGreedy
import utils
import argparse
from joblib import Parallel, delayed
import datetime

prior_eigen = None

def unpack(params):
    """Unpacks a parameter vector into c, mu, L"""
    c = params[:C]
    mus = params[C:C + C * K]
    mu = mus.reshape(C, K)
    Ls = params[C + C * K:]
    L = Ls.reshape(C, K, K)
    return c, mu, L


def pack(c, mu, L):
    """Packs c and mu and L into a parameter vector"""
    params = np.concatenate((c, mu.flatten(), L.flatten()), axis=0)
    return params


def clip(params):
    """Makes sure the Cholesky factor L is well-defined"""
    c, mu, L = unpack(params)
    # TODO implement more efficiently?
    for i in range(C):
        mask = np.logical_and(L[i, :, :] < cholesky_clip, np.eye(K, dtype=bool))
        L[i, :, :][mask] = cholesky_clip
        L[i, :, :][np.triu_indices(K, 1)] = 0
    return pack(c, mu, L)


def sample_posterior(params):
    """Samples a Q function from the posterior distribution"""
    c, mu, L = unpack(params)
    cluster = np.random.choice(np.arange(C), p=c)
    return np.dot(L[cluster, : , :], np.random.randn(K,)) + mu[cluster, :]


def normal_KL(c, mu, Sigma, c_bar, mu_bar, Sigma_bar, precision=None):
    """ Computes the KL between normals for two GMMs """

    global prior_eigen

    Sigma_bar_inv = np.linalg.inv(Sigma_bar) if precision is None else precision
    inv_b = Sigma_bar_inv[np.newaxis]
    if prior_eigen is None:
        prior_eigen, _ = np.linalg.eig(Sigma_bar[np.newaxis])

    posterior_eigen, _ = np.linalg.eig(Sigma[:, np.newaxis])
    posterior_eigen = np.real(posterior_eigen)

    mu_diff = mu[:, np.newaxis] - mu_bar[np.newaxis]

    return 0.5 * (np.sum(np.log(prior_eigen/posterior_eigen) + posterior_eigen/prior_eigen, axis=2) + \
                  np.matmul(np.matmul(mu_diff[:, :, np.newaxis], inv_b), mu_diff[:, :, :, np.newaxis])[:,:,0,0]- mu.shape[1])


def tight_ukl(c, mu, Sigma, c_bar, mu_bar, Sigma_bar, phi, psi, eps=0.01, max_iter=100):
    """ Solves Variational problem to tight the upper bound"""

    kl = normal_KL(c, mu, Sigma, c_bar, mu_bar, Sigma_bar)

    # Constants
    e_kl = np.exp(-kl) + 1e-10

    i = 0
    done = False
    while i < max_iter and not done:

        psi_bar = c_bar[np.newaxis] * phi / np.sum(phi, axis=0, keepdims=True)
        done = np.max(np.abs(psi - psi_bar)) < eps
        psi = psi_bar

        phi_bar = c[:, np.newaxis] * psi * e_kl / np.sum(psi * e_kl, axis=1, keepdims=True)
        done = done and np.max(np.abs(phi-phi_bar)) < eps
        phi = phi_bar

        i += 1

    return phi, psi


def UKL(c, mu, Sigma, c_bar, mu_bar, Sigma_bar, phi, psi, precision=None):
    """
    :param c:
    :param mu:
    :param Sigma:
    :param c_bar:
    :param mu_bar:
    :param Sigma_bar:
    :param phi: np.ndarray (posterior dim, prior dim)
    :param psi: np.ndarray (posterior dim, prior dim)
    :return:
    """
    kl = normal_KL(c, mu, Sigma, c_bar, mu_bar, Sigma_bar, precision=precision)   # (posterior, prior)

    kl_var = np.sum(phi * np.log(phi/psi))
    kl_gaussian = np.sum(phi * kl)

    return kl_var + kl_gaussian


def sample_gmm(n_samples, c, mu, L):
    """ Samples a mixture of Gaussians """
    vs = np.random.randn(n_samples, mu.shape[1])
    clusters = np.random.choice(np.arange(C), n_samples, p=c)
    ws = np.matmul(vs[:,np.newaxis], np.transpose(L[clusters], (0,2,1)))[:,:,0] + mu[clusters]
    return ws, vs


def objective(samples, params, Q, c_bar, mu_bar, Sigma_bar, operator, n_samples, phi, psi, precision=None):
    """Computes the negative ELBO"""
    c, mu, L = unpack(params)
    assert mu.shape == (C,K) and L.shape == (C,K,K)
    # We add a small constant to make sure Sigma is always positive definite
    Sigma = np.matmul(L, np.transpose(L, (0,2,1))) + np.eye(K)[np.newaxis] * 0.01
    assert Sigma.shape == (C,K,K) and np.all(np.linalg.eigvals(Sigma) > 0)
    weights, _ = sample_gmm(n_weights, c, mu, L)
    assert weights.shape == (n_weights,K)
    likelihood = operator.expected_bellman_error(Q, samples, weights)
    assert likelihood >= 0
    kl = UKL(c, mu, Sigma, c_bar, mu_bar, Sigma_bar, phi, psi, precision=precision)
    assert kl >= 0
    return likelihood + lambda_ * kl / n_samples


def gradient_KL(c, mu, L, c_bar, mu_bar, Sigma_bar, phi, psi, precision=None, tight_bound=True):
    if tight_bound:
        Sigma = np.matmul(L, np.transpose(L, (0,2,1)))
        psi = c[:, np.newaxis] * c_bar[np.newaxis]
        phi = np.array(psi)
        phi, psi = tight_ukl(c, mu, Sigma, c_bar, mu_bar, Sigma_bar, phi, psi, max_iter=max_iter_ukl)

    assert np.all(np.logical_and(phi >= 0, psi >= 0))

    Sigma_bar_inv = precision if precision is not None else np.linalg.inv(Sigma_bar)
    mu_diff = mu[:, np.newaxis] - mu_bar[np.newaxis]
    phi_m = np.argmax(phi, axis=1)
    grad_mu = np.sum(phi[:,:, np.newaxis, np.newaxis] * np.matmul(Sigma_bar_inv[np.newaxis], mu_diff[:,:,:, np.newaxis]), axis=1)[:,:,0]
    # grad_mu = np.squeeze(np.matmul(Sigma_bar_inv[np.newaxis], mu_diff[:, :, :, np.newaxis]))
    # grad_mu = grad_mu[np.arange(phi.shape[0]), phi_m]

    grad_L =  np.sum(phi[:,:, np.newaxis, np.newaxis] * (np.matmul(Sigma_bar_inv[np.newaxis], L[:, np.newaxis]) - np.linalg.inv(np.transpose(L, (0,2,1))[:, np.newaxis])), axis=1)
    assert np.all(np.logical_not(np.isnan(grad_L)))
    # grad_L = np.matmul(Sigma_bar_inv[np.newaxis], L[:, np.newaxis]) - np.linalg.inv(L[:, np.newaxis])
    # grad_L = grad_L[np.arange(phi.shape[0]), phi_m]
    grad_c = np.zeros(C)

    return grad_c, grad_mu, grad_L, phi, psi

def init_posterior(c, mu, L, c_bar, mu_bar, Sigma_bar, phi, psi, precision=None, max_iter=10000, eta=1e-5, eps=0.000001):
    i = 0
    Sigma = np.matmul(L, np.transpose(L, axes=(0, 2, 1)))
    ukl_prev = UKL(c, mu, Sigma, c_bar, mu_bar, Sigma_bar, phi, psi)
    done = False
    params = pack(c, mu, L)
    while not done and i < max_iter:
        if i % 100 == 0:
            grad_c, grad_mu, grad_L, phi, psi = gradient_KL(c, mu, L, c_bar, mu_bar, Sigma_bar, phi, psi, precision=precision)
        else:
            grad_c, grad_mu, grad_L, phi, psi = gradient_KL(c, mu, L, c_bar, mu_bar, Sigma_bar, phi, psi,
                                                            precision=precision, tight_bound=False)
        params = clip(params - eta * pack(grad_c, grad_mu, grad_L))
        c, mu, L = unpack(params)
        Sigma = np.matmul(L, np.transpose(L, axes=(0, 2, 1)))
        ukl = UKL(c, mu, Sigma, c_bar, mu_bar, Sigma_bar, phi, psi)
        done = np.abs(ukl-ukl_prev) < eps
        ukl_prev = ukl
        i += 1
        print("Initializing prior %d... UKL: %f" % (i, ukl))
    return params, phi, psi


def gradient(samples, params, Q, c_bar, mu_bar, Sigma_bar, operator, n_samples, phi, psi, precision=None):
    """Computes the objective function gradient"""
    c, mu, L = unpack(params)
    assert mu.shape == (C,K) and L.shape == (C,K,K)
    grad_c = np.zeros(c.shape)

    _, vs = utils.sample_mvn(n_weights * C, mu[0, :], L[0, :, :])

    ws = np.matmul(vs.reshape(C,n_weights,K), np.transpose(L, (0,2,1))) + mu[:, np.newaxis]
    assert ws.shape == (C, n_weights, K)
    be_grad = operator.gradient_be(Q, samples, ws.reshape(C*n_weights, K)).reshape(C, n_weights, K)
    assert be_grad.shape == (C, n_weights, K)
    # Gradient of the expected Bellman error wrt mu
    ebe_grad_mu = np.average(be_grad, axis=1)
    assert ebe_grad_mu.shape == (C,K)
    # Gradient of the expected Bellman error wrt L.
    ebe_grad_L = np.average(np.matmul(be_grad[:, :, :, np.newaxis], vs.reshape(C, n_weights, K)[:,:,np.newaxis]), axis=1)
    assert ebe_grad_L.shape == (C,K,K)
    ebe_grad_mu = c[:, np.newaxis] * ebe_grad_mu
    ebe_grad_L = c[:, np.newaxis, np.newaxis] * ebe_grad_L

    kl_grad_c, kl_grad_mu, kl_grad_L, phi, psi = gradient_KL(c, mu, L, c_bar, mu_bar, Sigma_bar, phi, psi, precision=precision)
    assert kl_grad_mu.shape == (C, K) and kl_grad_L.shape == (C,K,K)
    grad_mu = ebe_grad_mu + lambda_ * kl_grad_mu / n_samples
    assert grad_mu.shape == (C,K)
    grad_L = ebe_grad_L + lambda_ * kl_grad_L / n_samples
    assert grad_L.shape == (C,K,K)

    return pack(grad_c, grad_mu, grad_L)


def run(mdp, seed=None):

    if seed is not None:
        np.random.seed(seed)

    # Build the features
    features = build_features_gw(gw_size, n_basis, n_actions, state_dim, action_dim)
    # Create BellmanOperator
    operator = MellowBellmanOperator(kappa, tau, xi, gamma, state_dim, action_dim)
    # Create Q Function
    Q = LinearQFunction(features, np.arange(n_actions), state_dim, action_dim)
    # Initialize policies
    pi_u = EpsilonGreedy(Q, Q.actions, epsilon=1)
    pi_g = EpsilonGreedy(Q, Q.actions, epsilon=0)

    # Add random episodes if needed
    dataset = utils.generate_episodes(mdp, pi_u, n_episodes=random_episodes) if random_episodes > 0 else None
    n_init_samples = dataset.shape[0] if dataset is not None else 0

    # Load weights and construct prior distribution
    weights = utils.load_object(source_file)
    ws = np.array([w[1] for w in weights])
    np.random.shuffle(ws)
    # Take only the first n_source weights
    ws = ws[:n_source, :]
    mu_bar = ws
    Sigma_bar = np.tile(np.eye(K) * bw, (n_source,1,1))
    # We use higher regularization for the prior to prevent the ELBO from diverging
    Sigma_bar_inv = np.tile(((1/bw + sigma_reg) * np.eye(K))[np.newaxis], (n_source, 1, 1))
    c_bar = np.ones(n_source)/n_source


    # We initialize the parameters of the posterior to the best approximation of the posterior family to the prior
    c = np.ones(C) / C
    psi = c[:, np.newaxis] * c_bar[np.newaxis]
    phi = np.array(psi)

    mu = np.array([np.random.randn(K) + 100 * np.random.randn(K) for _ in range(C)])
    Sigma = np.array([np.eye(K) * (1 + cholesky_clip) for _ in range(C)])

    phi, psi = tight_ukl(c, mu, Sigma, c_bar, mu_bar, Sigma_bar, phi, psi, max_iter=max_iter_ukl)
    params, phi, psi = init_posterior(c, mu, Sigma, c_bar, mu_bar, Sigma_bar, phi, psi, max_iter=max_iter_ukl * 10, precision=Sigma_bar_inv)

    # Results
    iterations = []
    episodes = []
    n_samples = []
    evaluation_rewards = []
    learning_rewards = []
    episode_rewards = [0.0]
    l_2 = []
    l_inf = []
    sft = []
    fvals = []

    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0

    # Init env
    s = mdp.reset()
    h = 0
    Q._w = sample_posterior(params)

    # Learning
    for i in range(max_iter):

        # If we do not use time coherent exploration, resample parameters
        Q._w = sample_posterior(params) if not time_coherent else Q._w
        # Take greedy action wrt current Q-function
        a = np.array([np.argmax(Q.value_actions(s))])
        # Step
        s_prime, r, done, _ = mdp.step(a)
        # Build the new sample and add it to the dataset
        dataset = utils.add_sample(dataset, buffer_size, h, s, a, r, s_prime, done)

        # Take a step of gradient if needed
        if i % train_freq == 0:
            # Shuffle the dataset
            np.random.shuffle(dataset)
            # Estimate gradient
            g = gradient(dataset[:batch_size, :], params, Q, c_bar, mu_bar, Sigma_bar, operator, n_init_samples + i + 1, phi, psi, precision=Sigma_bar_inv)
            # Take a gradient step
            params, t, m_t, v_t = utils.adam(params, g, t, m_t, v_t, alpha=alpha)
            # Clip parameters
            params = clip(params)


        # c, mu, L = unpack(params)
        # Sigma = np.matmul(L, np.transpose(L, axes=(0, 2, 1)))
        # print(UKL(c, mu, Sigma, c_bar, mu_bar, Sigma_bar, phi, psi, precision=Sigma_bar_inv))

        # Add reward to last episode
        episode_rewards[-1] += r * gamma ** h

        s = s_prime
        h += 1
        if done or h >= mdp.horizon:

            episode_rewards.append(0.0)
            s = mdp.reset()
            h = 0
            Q._w = sample_posterior(params)

        # Evaluate model
        if i % eval_freq == 0:

            #Save current weights
            current_w = np.array(Q._w)

            # Evaluate MAP Q-function
            c, mu, _ = unpack(params)
            rew = 0
            for j in range(C):
                Q._w = mu[j]
                utils.plot_Q(Q)
                rew += utils.evaluate_policy(mdp, pi_g, render=render, initial_states=eval_states)[0]

            rew /= C

            learning_rew = np.mean(episode_rewards[-mean_episodes - 1:-1]) if len(episode_rewards) > 1 else 0.0
            br = operator.bellman_residual(Q, dataset) ** 2
            l_2_err = np.average(br)
            l_inf_err = np.max(br)
            sft_err = np.sum(utils.softmax(br, tau) * br)
            fval = objective(dataset, params, Q, c_bar, mu_bar, Sigma_bar, operator, n_init_samples + i + 1, \
                             phi, psi, precision=Sigma_bar_inv)

            # Append results
            iterations.append(i)
            episodes.append(len(episode_rewards) - 1)
            n_samples.append(n_init_samples + i + 1)
            evaluation_rewards.append(rew)
            learning_rewards.append(learning_rew)
            l_2.append(l_2_err)
            l_inf.append(l_inf_err)
            sft.append(sft_err)
            fvals.append(fval)

            # Make sure we restart from s
            mdp.reset(s)

            # Restore weights
            Q._w = current_w

            if verbose:
                print("Iter {} Episodes {} Rew(G) {} Rew(L) {} Fval {} L2 {} L_inf {} Sft {}".format(
                    i, episodes[-1], rew, learning_rew, fval, l_2_err, l_inf_err, sft_err))

    run_info = [iterations, episodes, n_samples, learning_rewards, evaluation_rewards, l_2, l_inf, sft, fval]
    weights = np.array(mu)

    return [mdp.door_x, weights, run_info]


# Global parameters
gamma = 0.99
n_actions = 4
state_dim = 2
action_dim = 1
max_iter_ukl = 60
render = False
verbose = True

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--kappa", default=100.)
parser.add_argument("--xi", default=0.5)
parser.add_argument("--tau", default=0.0)
parser.add_argument("--batch_size", default=100)
parser.add_argument("--max_iter", default=5000)
parser.add_argument("--buffer_size", default=10000)
parser.add_argument("--random_episodes", default=0)
parser.add_argument("--train_freq", default=1)
parser.add_argument("--eval_freq", default=50)
parser.add_argument("--mean_episodes", default=50)
parser.add_argument("--alpha", default=0.001)
parser.add_argument("--lambda_", default=0.000001)
parser.add_argument("--time_coherent", default=True)
parser.add_argument("--n_weights", default=200)
parser.add_argument("--n_source", default=10)
parser.add_argument("--env", default="two-room-gw")
parser.add_argument("--gw_size", default=5)
parser.add_argument("--sigma_reg", default=0.01)
parser.add_argument("--cholesky_clip", default=0.01)
# Door at -1 means random positions over all runs
parser.add_argument("--door", default=1.)
parser.add_argument("--door2", default=-1)
parser.add_argument("--n_basis", default=6)
parser.add_argument("--n_jobs", default=1)
parser.add_argument("--n_runs", default=1)
parser.add_argument("--file_name", default="gvt_{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
parser.add_argument("--source_file", default="source_tasks/gw5x5")
parser.add_argument("--bandwidth", default=.00001)     # Bandwidth for the Kernel Estimator
parser.add_argument("--post_components", default=1) # number of components of the posterior family


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
alpha = float(args.alpha)
lambda_ = float(args.lambda_)
time_coherent = bool(args.time_coherent)
n_weights = int(args.n_weights)
n_source = int(args.n_source)
sigma_reg = float(args.sigma_reg)
cholesky_clip = float(args.cholesky_clip)
env = str(args.env)
gw_size = int(args.gw_size)
door = float(args.door)
door2 = float(args.door2)
n_basis = int(args.n_basis)
n_jobs = int(args.n_jobs)
n_runs = int(args.n_runs)
file_name = str(args.file_name)
source_file = str(args.source_file)

# Number of features
K = n_basis ** 2 * n_actions
C = int(args.post_components)
bw = float(args.bandwidth)

# Generate tasks
doors = [np.random.uniform(0.5, gw_size - 0.5) if door < 0 else door for _ in range(n_runs)]
doors2 = [np.random.uniform(0.5, gw_size - 0.5) if door2 < 0 else door2 for _ in range(n_runs)]
if env == "two-room-gw":
    mdps = [WalledGridworld(np.array([gw_size, gw_size]), door_x=d) for d in doors]
elif env == "three-room-gw":
    mdps = [MarcellosGridworld(np.array([gw_size, gw_size]), door_x=(d1,d2)) for (d1,d2) in zip(doors,doors2)]
eval_states = [np.array([0., 0.]) for _ in range(10)]

if n_jobs == 1:
    results = [run(mdp) for mdp in mdps]
elif n_jobs > 1:
    seeds = [np.random.randint(1000000) for _ in range(n_runs)]
    results = Parallel(n_jobs=n_jobs)(delayed(run)(mdp,seed) for (mdp,seed) in zip(mdps,seeds))

utils.save_object(results, file_name)


