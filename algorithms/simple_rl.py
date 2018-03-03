import numpy as np
import envs.walled_gridworld as env
import utils
import algorithms.e_greedy_policy as policy
import algorithms.linear_q_function as Q
import algorithms.regularized_lsvi as lsvi
import features.grbf as grbf
"""
Implementation of a simple RL algorithm based on an e-greedy policy and a linearly parameterized Q function
"""
def simple_RL(mdp, Q, epsilon=0, K=10, batch_size=1):
    pol = policy.eGreedyPolicy(Q, Q.actions, epsilon)
    for i in range(K):
        samples = utils.generate_episodes(mdp, pol, batch_size)
        targets = Q.compute_bellman_target(samples[:, 1:])
        w = lsvi.RegularizedLSVI.solve(Q.compute_features(samples), targets, prior=False)
        Q.update_weights(w)


if __name__ == '__main__':
    n = 3
    acts = 4
    k = n*n*acts
    world = env.WalledGridworld(np.array((n, n)))
    mean = np.array([[x, y, a] for x in range(n) for y in range(n) for a in range(acts)])
    variance = np.ones(k)
    features = grbf.GaussianRBF(mean, variance, K=k, dims=world.state_dim+world.action_dim)
    q = Q.LinearQFunction(range(acts), features, np.zeros(k), state_dim=world.state_dim, action_dim=world.action_dim)

    simple_RL(world, q)