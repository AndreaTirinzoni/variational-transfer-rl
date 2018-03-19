import numpy as np
import pickle
from time import sleep


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
    s = data[:, 1:a_idx].squeeze()
    a = data[:, a_idx:r_idx].squeeze()
    r = data[:, r_idx]
    s_prime = data[:, s_idx:-1].squeeze()
    absorbing = data[:, -1]
    sa = data[:, 1:r_idx]

    return t, s, a, r, s_prime, absorbing, sa


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