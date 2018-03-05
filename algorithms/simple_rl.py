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
    samples = utils.generate_episodes(mdp, pol, batch_size, render=render)
    feat = Q.compute_features(samples[:, 1:])
    for i in range(K):
        new_samples = utils.generate_episodes(mdp, pol, batch_size, render=render)
        samples = np.vstack((samples, new_samples))
        targets = Q.compute_bellman_target(samples[:, 1:])
        feat = np.vstack((feat, Q.compute_features(new_samples[:, 1:])))
        w = lsvi.RegularizedLSVI.solve(feat, targets, prior=False)
        Q.update_weights(w)
        rew = np.sum(new_samples[:, 1+mdp.state_dim+mdp.action_dim])/batch_size
        r.append(rew)
        if verbose:
            print("Iteration " + str(i))
            print("Reward: " + str(rew))
    if render:
        mdp._render(close=True)
    return r


if __name__ == '__main__':
    n = 5
    acts = 4
    k = n*n*acts
    world = env.WalledGridworld(np.array((n, n)))
    mean = np.array([[x, y, a] for x in range(0, n) for y in range(0, n) for a in range(acts)])
    variance = (np.ones(k)/3)**2
    features = grbf.GaussianRBF(mean, variance, K=k, dims=world.state_dim+world.action_dim)
    q = Q.LinearQFunction(range(acts), features, np.zeros(k), state_dim=world.state_dim, action_dim=world.action_dim)
    episodes = 50
    r = simple_RL(world, q, epsilon=0.05, K=episodes, render=True, verbose=True, batch_size=1)
    plt.plot(np.arange(episodes), np.asarray(r))
    plt.show()