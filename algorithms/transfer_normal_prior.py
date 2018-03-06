import numpy as np
import envs.walled_gridworld as wgw
import algorithms.linear_q_function as Q
import features.grbf as grbf
import algorithms.simple_rl as rl
import utils


def transfer_rl(mdp, Q, prior_parameters, epsilon=0, K=1, batch_size=1, verbose=False, render=False, n_fit=20):

    # Sample weights to initialize Q
    w = list()
    for a in Q.actions:
        # w.append(np.random.multivariate_normal(prior_parameters[0][a], prior_parameters[1][a]))
        w.append(prior_parameters[0][a])
    print(prior_parameters[0][0])
    print(prior_parameters[1][0])
    Q.update_weights(w)

    # Run RL
    return rl.simple_RL(mdp, Q, epsilon, K, batch_size, render, verbose, n_fit, prior_parameters)


"""
Solves the Walled Gridworld for 5 different door positions and uses the induced prior over Q functions 
to solve a random tasks from the sources.
"""
if __name__ == '__main__':

    #Initial parameters
    grid_size = 5
    k = (grid_size+1)**2
    worlds = list()
    q_functions = list()
    n_acts = 4

    #Load saved

    prior_means, prior_variances = utils.load_object("prior_information_" + str(grid_size))

    # Create features for the gridworld
    mean = np.array([[x, y] for x in range(grid_size+1) for y in range(grid_size+1)])
    variance = (np.ones(k)/2)**2
    feat = grbf.GaussianRBF(mean, variance, K=k)

    # Create Tasks (Environments)
    for pos in range(1, grid_size+1):
        w = wgw.WalledGridworld(np.asarray((grid_size, grid_size)), door_x=pos)
        worlds.append(w)
        q_functions.append(Q.LinearQFunction(range(n_acts), feat, params=np.zeros(k), state_dim=w.state_dim, action_dim=w.action_dim))

    # # Solve the Tasks
    # for i in range(len(worlds)):
    #     print("Solving World with port at " + str(i + 1))
    #     rl.simple_RL(worlds[i], q_functions[i], epsilon=0.05, K=30, verbose=True, render=False)
    #
    # # Set induced prior parameters
    # prior_means = list()
    # prior_variances = list()
    # weights = [q._w for q in q_functions]
    # for a in range(n_acts):
    #     m = sum([w[a] for w in weights])/len(weights)
    #     covar = sum(map(lambda w: w.reshape(k,1) @ w.reshape(1,k), [w[a]-m for w in weights]))/len(weights)
    #     prior_means.append(m)
    #     prior_variances.append(covar)


    # Save prior information
    utils.save_object((prior_means, prior_variances), "prior_information_" + str(grid_size))


    # Solve new task
    q = Q.LinearQFunction(range(n_acts), feat, params=np.zeros(k), state_dim=w.state_dim, action_dim=w.action_dim)
    r = transfer_rl(worlds[np.random.randint(0, len(worlds))], q, (prior_means, prior_variances), batch_size=10, epsilon=0.01, K=20, verbose=True, render=True, n_fit=10)

    print(r)