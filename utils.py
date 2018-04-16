import numpy as np
import pickle
from time import sleep
import matplotlib.pyplot as plt


def evaluate_policy(mdp, policy, criterion='discounted', n_episodes=1, initial_states=None, render=False):
    """
    Evaluates a policy on a given MDP.

    Parameters
    ----------
    mdp: the environment to use in the evaluation
    policy: the policy to evaluate
    criterion: either 'discounted' or 'average'
    n_episodes: the number of episodes to generate in the evaluation
    initial_states: either None (i), a numpy array (ii), or a list of numpy arrays (iii)
      - (i) initial states are drawn from the MDP distribution
      - (ii) the given array is used as initial state for all episodes
      - (iii) n_episodes is ignored and the episodes are defined by their initial states

    Returns
    -------
    The mean score, their standard deviation, the mean length of each episode, and their standard deviation
    """

    assert criterion == 'average' or criterion == 'discounted'

    if initial_states is None or type(initial_states) is np.ndarray:
        scores = [_single_eval(mdp, policy, criterion, initial_states, render) for _ in range(n_episodes)]
    elif type(initial_states) is list:
        scores = [_single_eval(mdp, policy, criterion, init_state, render) for init_state in initial_states]

    scores = np.array(scores)
    return np.mean(scores[:, 0]), np.std(scores[:, 0]), np.mean(scores[:, 1]), np.std(scores[:, 1])


def _single_eval(mdp, policy, criterion, initial_state, render):

    score = 0
    gamma = mdp.gamma if criterion == "discounted" else 1

    s = mdp.reset(initial_state)
    t = 0

    while t < mdp.horizon:

        a = policy.sample_action(s)
        if render:
            mdp._render(a=a)
            sleep(0.01)
        s, r, done, _ = mdp.step(a)
        score += r * gamma ** t
        t += 1
        if done:
            break

    return score if criterion == "discounted" else score / t, t


def generate_episodes(mdp, policy, n_episodes=1, render=False):
    """
    Generates episodes in a given mdp using a given policy

    Parameters
    ----------
    mdp: the environment to use
    policy: the policy to use
    n_episodes: the number of episodes to generate

    Returns
    -------
    A matrix where each row corresponds to a single sample (t,s,a,r,s',absorbing)
    """

    episodes = [_single_episode(mdp, policy, render) for _ in range(n_episodes)]

    return np.concatenate(episodes)


def _single_episode(mdp, policy, render=False):
    episode = np.zeros((mdp.horizon, 1 + mdp.state_dim + mdp.action_dim + 1 + mdp.state_dim + 1))
    a_idx = 1 + mdp.state_dim
    r_idx = a_idx + mdp.action_dim
    s_idx = r_idx + 1

    s = mdp.reset()
    t = 0

    while t < mdp.horizon:

        episode[t, 0] = t
        episode[t, 1:a_idx] = s

        a = policy.sample_action(s)
        s, r, done, _ = mdp.step(a)
        if render:
            mdp._render(a=a)
        episode[t, a_idx:r_idx] = a
        episode[t, r_idx] = r
        episode[t, s_idx:-1] = s
        episode[t, -1] = 1 if done else 0

        t += 1
        if done:
            break

    return episode[0:t, :]


def split_data(data, state_dim, action_dim):
    """
    Splits the data into (t,s,a,r,s_prime,absorbing,sa)
    """

    assert data.shape[1] == 3 + 2 * state_dim + action_dim

    a_idx = 1 + state_dim
    r_idx = a_idx + action_dim
    s_idx = r_idx + 1

    t = data[:, 0]
    s = data[:, 1:a_idx]
    a = data[:, a_idx:r_idx]
    r = data[:, r_idx]
    s_prime = data[:, s_idx:-1]
    absorbing = data[:, -1]
    sa = data[:, 1:r_idx]

    return t, s, a, r, s_prime, absorbing, sa


def add_sample(dataset, capacity, t, s, a, r, s_prime, absorbing):
    """Adds sample to a dataset of limited capacity"""
    sample = np.concatenate([np.array([t]), s, a, np.array([r]), s_prime, np.array([1 if absorbing else 0])])[np.newaxis, :]
    if dataset is None:
        return sample
    dataset = np.concatenate((dataset, sample), axis=0)
    return dataset if dataset.shape[0] <= capacity else dataset[1:, :]


def save_object(obj, file_name):
    """
    Saves an object to a file in .pkl format

    Parameters
    ----------
    file_name: the file where to save (without extension)
    """
    with open(file_name + ".pkl", 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def load_object(file_name):
    """
    Loads an object from a file in .pkl format

    Parameters
    ----------
    file_name: the file where to load (without extension)

    Returns
    -------
    The loaded object
    """

    with open(file_name + ".pkl", 'rb') as file:
        return pickle.load(file)


def plot_Q(Q, size=(5,5)):

    V = [[], [], [], []]
    Vmax = []
    pi = []
    X = np.arange(0.0, size[0]+.1, 0.1)
    Y = np.arange(0.0, size[1]+.1, 0.1)

    for x in X:
        for y in Y:
            vals = Q.value_actions(np.array([y, x]))
            Vmax.append(np.max(vals))
            pi.append(np.argmax(vals))
            vals = vals.reshape(4,1)
            V[0].append(vals[0])
            V[1].append(vals[1])
            V[2].append(vals[2])
            V[3].append(vals[3])

    for v in V:
        v[-1][0] = 0.0
    Vmax[-1] = 0.0
    V = [np.flip(np.array(v).reshape(X.size, X.size), axis=0) for v in V]

    fig, ax = plt.subplots(3, 2)
    f00 = ax[0, 0].imshow(V[0], cmap="hot", interpolation="gaussian")
    ax[0, 0].set_title("UP (black)")
    ax[0, 0].axes.get_yaxis().set_visible(False)
    ax[0, 0].axes.get_xaxis().set_visible(False)
    f01 = ax[0, 1].imshow(V[1], cmap="hot", interpolation="gaussian")
    ax[0, 1].set_title("RIGHT (red)")
    ax[0, 1].axes.get_yaxis().set_visible(False)
    ax[0, 1].axes.get_xaxis().set_visible(False)
    f10 = ax[1, 0].imshow(V[2], cmap="hot", interpolation="gaussian")
    ax[1, 0].set_title("DOWN (orange)")
    ax[1, 0].axes.get_yaxis().set_visible(False)
    ax[1, 0].axes.get_xaxis().set_visible(False)
    f11 = ax[1, 1].imshow(V[3], cmap="hot", interpolation="gaussian")
    ax[1, 1].set_title("LEFT (white)")
    ax[1, 1].axes.get_yaxis().set_visible(False)
    ax[1, 1].axes.get_xaxis().set_visible(False)
    f20 = ax[2, 0].imshow(np.flip(np.array(Vmax).reshape(X.size, Y.size), axis=0), cmap="hot", interpolation="gaussian")
    ax[2, 0].set_title("Value Function")
    ax[2, 0].axes.get_yaxis().set_visible(False)
    ax[2, 0].axes.get_xaxis().set_visible(False)
    f21 = ax[2, 1].imshow(np.flip(np.array(pi).reshape(X.size, Y.size), axis=0), cmap="hot", interpolation="nearest")
    ax[2, 1].set_title("Policy")
    ax[2, 1].axes.get_yaxis().set_visible(False)
    ax[2, 1].axes.get_xaxis().set_visible(False)

    fig.colorbar(f00, ax=ax[0, 0])
    fig.colorbar(f01, ax=ax[0, 1])
    fig.colorbar(f10, ax=ax[1, 0])
    fig.colorbar(f11, ax=ax[1, 1])
    fig.colorbar(f20, ax=ax[2, 0])
    fig.colorbar(f21, ax=ax[2, 1])
    plt.show()


def mellow_max(a, kappa, axis=-1):
    """
    Computes the mellow-max of array a along the given axis
    :param a: a numpy array
    :param kappa: the (inverse) temperature
    :param axis: the axis along which the mellow-max is computed. By default, the last axis is used
    :return: a numpy array. The axis along which mellow-max is computed is always removed
    """
    mx = np.max(a, axis=axis, keepdims=True)
    mm_exp = np.exp(kappa * (a - mx))
    mm_sum = np.sum(mm_exp, axis=axis, keepdims=True)
    return np.squeeze(np.log(mm_sum / a.shape[axis]) / kappa + mx, axis=axis)


def softmax(a, tau, axis=-1):
    """
    Computes the softmax of array a along the given axis
    :param a: a numpy array
    :param tau: the (inverse) temperature
    :param axis: the axis along which the softmax is computed. By default, the last axis is used
    :return: a numpy array with the same shape of a
    """
    mx = np.max(a, axis=axis, keepdims=True)
    num = np.exp(tau * (a - mx))
    return num / np.sum(num, axis=axis, keepdims=True)


def adam(params, grad, t, m_t, v_t, alpha=0.001, beta_1=0.9, beta_2=0.999, eps=1e-8):
    """
    Applies a gradient step to the given parameters based on ADAM update rule
    :param params: a numpy array of parameters
    :param grad: the objective function gradient evaluated in params. This must have the same shape of params
    :param t: the iteration number
    :param m_t: first order momentum
    :param v_t: second order momentum
    :param alpha: base learning rate
    :param beta_1: decay of first order momentum
    :param beta_2: decay of second order momentum
    :param eps: small constant
    :return: the updated parameters, iteration number, first order momentum, and second order momentum
    """
    t += 1
    m_t = beta_1 * m_t + (1 - beta_1) * grad
    v_t = beta_2 * v_t + (1 - beta_2) * grad ** 2
    m_t_hat = m_t / (1 - beta_1 ** t)
    v_t_hat = v_t / (1 - beta_2 ** t)
    return params - alpha * m_t_hat / (np.sqrt(v_t_hat) + eps), t, m_t, v_t


def KL(mu1, Sigma1, mu2, Sigma2, precision=True):
    """
    Computes the KL divergence between two multivariate Gaussian distributions.
    :param mu1: mean of the first Gaussian. A numpy array of shape (K,)
    :param Sigma1: covariance of the first Gaussian. A numpy array of shape (K,K)
    :param mu2: mean of the second Gaussian. A numpy array of shape (K,)
    :param Sigma2: covariance or precision of the second Gaussian. A numpy array of shape (K,K)
    :param precision: whether the precision of the second Gaussian or its covariance is passed
    :return: the KL divergence (scalar)
    """
    Sigma2_inv = Sigma2 if precision else np.linalg.inv(Sigma2)
    mu_diff = mu1 - mu2
    return (np.log(1 / (np.linalg.det(Sigma1) * np.linalg.det(Sigma2_inv))) +
            np.trace(np.dot(Sigma2_inv,Sigma1)) +
            np.dot(np.dot(mu_diff.T,Sigma2_inv),mu_diff) - mu1.shape[0]) / 2


def gradient_KL(mu1, L1, mu2, Sigma2, precision=True):
    """
    Computes the gradient of the KL between two multivariate Gaussians wrt mean and Cholensky factor of the first
    :param mu1: mean of the first Gaussian. A numpy array of shape (K,)
    :param L1: Cholensky factor of the first Gaussian's covariance. A numpy array of shape (K,K).
    :param mu2: mean of the second Gaussian. A numpy array of shape (K,)
    :param Sigma2: covariance or precision of the second Gaussian. A numpy array of shape (K,K)
    :param precision: whether the precision of the second Gaussian or its covariance is passed
    :return: the gradient wrt mu1 (K-dimensional array) and the gradient wrt L1 (KxK matrix)
    """
    Sigma2_inv = Sigma2 if precision else np.linalg.inv(Sigma2)
    grad_mu = np.dot(Sigma2_inv, mu1 - mu2)
    # If L1 is not invertible, we return 0 by default
    try:
        grad_L = np.dot(Sigma2_inv, L1) - np.linalg.inv(L1).T
    except np.linalg.LinAlgError:
        print("WARNING: L1 is not invertible")
        grad_L = 0.0
    return grad_mu, grad_L


def sample_mvn(n_samples, mu, L):
    """
    Fast sampling from a multivariate Gaussian using its Cholensky factor
    :param n_samples: the number of samples to draw
    :param mu: mean vector. A numpy array of shape (K,)
    :param L: Cholensky factor of the covariance. A numpy array of shape (K,K).
    :return: Two arrays of shape (n_samples,K): one with the mvn samples and one with the corresponding uvn samples
    """
    vs = np.random.randn(n_samples, mu.shape[0])
    ws = np.dot(vs, L.T) + mu
    return ws, vs
