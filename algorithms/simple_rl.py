import numpy as np
import envs.walled_gridworld as env
import utils
import algorithms.e_greedy_policy as policy
import algorithms.linear_q_function as Q
import algorithms.regularized_lsvi as lsvi
import features.grbf as grbf
import matplotlib.pyplot as plt

"""
Implementation of a simple RL algorithm based on an e-greedy policy and a linearly parameterized Q function
without the prior distribution.
"""
def simple_RL(mdp, Q, epsilon=0, K=1, batch_size=1, render=False, verbose=False):
    pol = policy.eGreedyPolicy(Q, Q.actions, epsilon)
    r = list()
    n_act = len(Q.actions)
    samples = _generate_episodes(mdp, pol, n_act, batch_size, render=render)
    feat = [Q.compute_features(samples[a][:, 1:]) for a in range(n_act)]

    for i in range(K):
        new_samples = _generate_episodes(mdp, pol, n_act, batch_size, render=render)
        samples = _stack(samples, new_samples)
        feat = _stack(feat, [Q.compute_features(new_samples[a][:, 1:]) for a in range(n_act)])

        for k in range(10):
            targets = [Q.compute_bellman_target(samples[a][:, 1:]) for a in range(n_act)]
            for a in range(n_act):
                if feat[a].shape[0] > 0:
                    w = lsvi.RegularizedLSVI.solve(feat[a], targets[a], prior=False)
                    Q.update_weights(w, a)


        rew = sum(map(lambda a: np.sum(a[:, 1+mdp.state_dim+mdp.action_dim]), new_samples))/batch_size
        r.append(rew)
        if verbose:
            print("Iteration " + str(i))
            print("Reward: " + str(rew))
    if render:
        mdp._render(close=True)
    return r


def _generate_episodes(mdp, policy, actions, n_episodes, render=False):
    samples = utils.generate_episodes(mdp, policy, n_episodes, render)
    split_samples = [None for a in range(actions)]
    action = mdp.state_dim + 1
    for a in range(actions):
        split_samples[a] = samples[np.where(samples[:, action: action+mdp.action_dim] == a)[0], :]
    return split_samples


def _stack(l1, l2):
    assert len(l1) == len(l2)
    l = list()
    for i in range(len(l1)):
        if l1[i].shape[0] > 0  and l2[i].shape[0] > 0:
            if l1[i].ndim == 2 and l2[i].ndim == 2:
                l.append(np.vstack((l1[i], l2[i])))
            elif l1[i].ndim == 1 and l2[i].ndim == 1:
                l.append(np.hstack((l1[i], l2[i])))
        elif l1[i].shape[0] > 0:
            l.append(l1[i])
        else:
            l.append(l2[i])
    return l


if __name__ == '__main__':
    n = 5
    acts = 4
    k = n*n
    world = env.WalledGridworld(np.array((n, n)), door_x=n/2)
    mean = np.array([[x, y] for x in range(0, n) for y in range(0, n)])
    variance = (np.ones(k)/3)**2
    features = grbf.GaussianRBF(mean, variance, K=k, dims=world.state_dim)
    q = Q.LinearQFunction(range(acts), features, np.zeros(k), state_dim=world.state_dim, action_dim=world.action_dim)
    episodes = 20
    r = simple_RL(world, q, epsilon=0.2, K=episodes, render=True, verbose=True, batch_size=1)

    print(q.compute_all_actions(np.array((5,5))))

    # plt.plot(np.arange(episodes), np.asarray(r))
    # plt.show()