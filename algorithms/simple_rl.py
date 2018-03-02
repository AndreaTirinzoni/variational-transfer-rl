import numpy as np
import envs.walled_gridworld as env
import utils
import algorithms.e_greedy_policy as policy
import algorithms.linear_q_function as Q
import algorithms.regularized_lsvi as lsvi
"""
Implementation of a simple RL algorithm based on an e-greedy policy and linearly parameterized Q function
"""
def simple_RL(mdp, Q, epsilon=0, K=10):
    pol = policy.eGreedyPolicy(Q, Q.actions, epsilon)
    for i in range(K):
        samples = utils.generate_episodes(mdp, pol, 100)
        _, s, a, r, s_prime, absorbing, _ = utils.split_data(samples, mdp.state_dim, mdp.action_dim)
        samples = np.hstack((s, a, r, s_prime, absorbing))
        targets = Q.compute_bellman_targets(samples)
        w = lsvi.RegularizedLSVI.solve(Q.compute_features(samples), targets, prior=False)
        Q.update_weights(w)


if __name__ == '__main__':
    