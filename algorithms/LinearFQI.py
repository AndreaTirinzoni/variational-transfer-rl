import numpy as np
import utils
from VariationalTransfer import BellmanOperator as bellmanop
from algorithms import e_greedy_policy as policy, regularized_lsvi as lsvi

"""
Implementation of linear FQI procedure with a unique regressor for Q function.
"""
def linearFQI(mdp, Q, epsilon=0, n_iter=1, batch_size=1, render=False, verbose=False, n_fit=20, bellman_operator=None, r_seed=None):

    if r_seed is not None:
        np.random.seed(r_seed)

    pol = policy.eGreedyPolicy(Q, Q.actions, epsilon)
    pol_g = policy.eGreedyPolicy(Q, Q.actions, 0)
    r = list()
    rew, _, _, _ = utils.evaluate_policy(mdp, pol_g, n_episodes=5, initial_states=np.array([0., 0.]), render=render)
    r.append(rew)
    if verbose:
        utils.plot_Q(Q, size=tuple(mdp.size))

    samples = utils.generate_episodes(mdp, pol, batch_size, render=False)
    feat = Q.compute_features(samples[:, 1:])

    if bellman_operator is None:
        bellman = bellmanop.BellmanOperator(Q)
    else:
        bellman_operator.set_Q(Q)
        bellman = bellman_operator

    for i in range(n_iter):
        new_samples = utils.generate_episodes(mdp, pol, batch_size, render=False)
        samples = np.vstack((samples, new_samples))
        feat = np.vstack((feat, Q.compute_features(new_samples[:, 1:])))

        for k in range(n_fit):
            targets = bellman(samples[:, 1:])
            w = lsvi.RegularizedLSVI.solve(feat, targets, prior=False)
            Q.update_weights(w)

        if render:
            mdp._render(close=True)

        rew, _, _, _ = utils.evaluate_policy(mdp, pol_g, n_episodes=5, initial_states=np.array([0., 0.]), render=render)
        r.append(rew)
        if verbose:
            print("===============================================")
            print("Iteration " + str(i))
            print("Reward: " + str(rew))
            print("===============================================")
            utils.plot_Q(Q, size=tuple(mdp.size))

    return r, Q