import numpy as np
import envs.walled_gridworld as wgw
import algorithms.linear_q_function as Q
import features.grbf as grbf
import algorithms.simple_rl as rl
import utils
import algorithms.transfer_normal_prior as transfer
import algorithms.simple_rl as rl

if __name__ == '__main__':

    # Initial parameters
    grid_size = 5
    k = (grid_size + 1) ** 2
    worlds = list()
    q_functions = list()
    n_acts = 4

    # Create features for the gridworld
    mean = np.array([[x, y] for x in range(grid_size + 1) for y in range(grid_size + 1)])
    variance = (np.ones(k) / 2) ** 2
    feat = grbf.GaussianRBF(mean, variance, K=k)

    # # Create Tasks (Environments)
    # for pos in range(1, grid_size + 1):
    #     w = wgw.WalledGridworld(np.asarray((grid_size, grid_size)), door_x=pos)
    #     worlds.append(w)
    #     q_functions.append(Q.LinearQFunction(range(n_acts), feat, params=np.zeros(k), state_dim=w.state_dim, action_dim=w.action_dim))

    worlds, _ = utils.load_object("source_tasks_5_walled_grid")


    # Permute order of the tasks
    # tasks = np.random.permutation(np.arange(0, len(worlds)))
    tasks = range(len(worlds))
    prior_mean = [np.zeros(k) for _ in range(n_acts)]
    prior_variances = [1e200*np.eye(k) for _ in range(n_acts)]

    # Run RL transferring knowledge from the past tasks
    weights = list()
    scores = list()
    for t in tasks:
        q = Q.LinearQFunction(range(n_acts), feat, params=np.zeros(k), state_dim=worlds[t].state_dim, action_dim=worlds[t].action_dim)
        if len(weights) > 1:
            r = transfer.transfer_rl(worlds[t], q, (prior_mean, prior_variances), epsilon=0.11, batch_size=10, K=10, verbose=True, render=True)
        else:
            r = rl.simple_RL(worlds[t], q, epsilon=0.2, K=10, batch_size=2, verbose=True, render=True)

        weights.append(q._w)
        scores.append(r)
        # Update mean&variance
        for a in range(n_acts):
            m = sum([w[a] for w in weights])/len(weights)
            covar = sum(map(lambda x: np.diag(x * x), [w[a]-m for w in weights]))/len(weights)
            prior_variances[a] = covar
            prior_mean[a] = m


