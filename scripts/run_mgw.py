import numpy as np
import matplotlib.pyplot as plt
from algorithms.simple_rl import simple_RL
import features.grbf as grbf
import envs.marcellos_gridworld as env
import algorithms.linear_q_function as Q
import algorithms.transfer_normal_prior as tr
import utils


"""
Test transfer with linear regressors and RBF in Marcello's Gridworld
"""
if __name__ == '__main__':
    n = 5
    acts = 4
    k = (n + 1) * (n + 1)
    door_1 = n/2
    episodes = 50
    saved_prior = True

    # Create features for the gridworld
    mean = np.array([[x, y] for x in range(n + 1) for y in range(n + 1)])
    variance = (np.ones(k) / 2) ** 2
    feat = grbf.GaussianRBF(mean, variance, K=k)

    if not saved_prior:
        # Create tasks
        worlds = list()
        q_functions = list()
        for door_2 in range(1, n + 1):
            w = env.MarcellosGridworld(np.asarray((n, n)), door_x=(door_1, door_2-0.5))
            worlds.append(w)
            q_functions.append(
                Q.LinearQFunction(range(acts), feat, params=np.zeros(k), state_dim=w.state_dim, action_dim=w.action_dim))

        # Solve the Tasks
        rew = list()
        for i in range(len(worlds)):
            print("Solving World with port at " + str(i + 1))
            r = simple_RL(worlds[i], q_functions[i], epsilon=0.2, K=episodes, batch_size=2, verbose=True, render=False)
            rew.append(np.array(r))

        rew = np.array(rew)
        # Set induced prior parameters
        prior_means = list()
        prior_variances = list()
        weights = [q._w for q in q_functions]
        for a in range(acts):
            m = sum([w[a] for w in weights]) / len(weights)
            # covar = sum(map(lambda x: x.reshape(k,1) @ x.reshape(1,k), [w[a]-m for w in weights]))/len(weights)
            covar = sum(map(lambda x: np.diag(x * x), [w[a] - m for w in weights])) / len(weights)
            prior_means.append(m)
            prior_variances.append(covar)

        utils.save_object((prior_means, prior_variances), "prior_information_" + str(n) + "_first_fixed")
        utils.save_object((worlds, q_functions, rew), "source_tasks_" + str(n) + "_marcellos_grid_first_fixed")

    else:
        _, q_functions, _ = utils.load_object("source_tasks_" + str(n) + "_marcellos_grid_first_fixed")
        _, q = utils.load_object("source_tasks_" + str(n) + "_marcellos_grid_second_fixed")
        q_functions += q

        # Set induced prior parameters
        prior_means = list()
        prior_variances = list()
        weights = [q._w for q in q_functions]
        for a in range(acts):
            m = sum([w[a] for w in weights]) / len(weights)
            # covar = sum(map(lambda x: x.reshape(k,1) @ x.reshape(1,k), [w[a]-m for w in weights]))/len(weights)
            covar = sum(map(lambda x: np.diag(x * x), [w[a] - m for w in weights])) / len(weights)
            prior_means.append(m)
            prior_variances.append(covar)

        prior_means, prior_variances = utils.load_object("prior_information_" + str(n))

    # Solve new task
    r = list()
    for _ in range(7):
        d2 = 0.5 #np.random.ranf(1)[0]*4 + .5
        w = env.MarcellosGridworld(np.asarray((n, n)), door_x=(door_1, d2))
        q = Q.LinearQFunction(range(acts), feat, params=np.zeros(k), state_dim=w.state_dim,
                              action_dim=w.action_dim)
        r.append(np.array(
            tr.transfer_rl(w, q, (prior_means, prior_variances), batch_size=10,
                        epsilon=0.1, K=10, verbose=True, render=True, n_fit=20)))

    # Plot performance

    rew = np.asarray(r)
    mean_r = np.mean(rew, axis=0)
    std_r = np.std(rew, axis=0)
    itr = np.arange(mean_r.size)

    plt.plot(itr, mean_r, label="Task not seen before", color='red')
    plt.fill_between(itr, mean_r + std_r, mean_r - std_r, color='red', alpha=0.3)
    plt.legend()
    plt.show()