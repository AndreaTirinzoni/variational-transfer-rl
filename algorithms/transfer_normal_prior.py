import numpy as np
import algorithms.linear_q_function as Q
import features.grbf as grbf
import algorithms.simple_rl as rl
import utils
import matplotlib.pyplot as plt
import os.path
import envs.walled_gridworld as wgw

def transfer_rl(mdp, Q, prior_parameters, epsilon=0, K=1, batch_size=1, verbose=False, render=False, n_fit=20):

    # Sample weights to initialize Q
    w = list()
    for a in Q.actions:
        w.append(np.random.multivariate_normal(prior_parameters[0][a], prior_parameters[1][a]))
        # w.append(prior_parameters[0][a])
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
    prior_loaded = False
    if os.path.isfile("prior_information_" + str(grid_size) + ".pkl"):
        prior_means, prior_variances = utils.load_object("prior_information_" + str(grid_size))
        prior_loaded = True


    # Create features for the gridworld
    mean = np.array([[x, y] for x in range(grid_size+1) for y in range(grid_size+1)])
    variance = (np.ones(k)/2)**2
    feat = grbf.GaussianRBF(mean, variance, K=k)


    sources_loaded = False
    if os.path.isfile("source_tasks_5_walled_grid.pkl"):
        worlds, q_functions = utils.load_object("source_tasks_5_walled_grid")
        sources_loaded = True



    if not sources_loaded:
        # Create Tasks (Environments)
        for pos in range(1, grid_size+1):
            w = wgw.WalledGridworld(np.asarray((grid_size, grid_size)), door_x=pos)
            worlds.append(w)
            q_functions.append(Q.LinearQFunction(range(n_acts), feat, params=np.zeros(k), state_dim=w.state_dim, action_dim=w.action_dim))

        # Solve the Tasks
        for i in range(len(worlds)):
            print("Solving World with port at " + str(i + 1))
            rl.simple_RL(worlds[i], q_functions[i], epsilon=0.2, K=20, batch_size=2, verbose=True, render=True)

    if not prior_loaded:
        # Set induced prior parameters
        prior_means = list()
        prior_variances = list()
        weights = [q._w for q in q_functions]
        for a in range(n_acts):
            m = sum([w[a] for w in weights])/len(weights)
            # covar = sum(map(lambda x: x.reshape(k,1) @ x.reshape(1,k), [w[a]-m for w in weights]))/len(weights)
            covar = sum(map(lambda x: np.diag(x * x), [w[a]-m for w in weights]))/len(weights)
            prior_means.append(m)
            prior_variances.append(covar)

    if not prior_loaded and not sources_loaded:
        # Save prior information
        utils.save_object((prior_means, prior_variances), "prior_information_" + str(grid_size))
        utils.save_object((worlds, q_functions), "source_tasks_" + str(grid_size) + "_walled_grid")


    # Solve new task
    r = list()
    for _ in range(len(worlds)):
        q = Q.LinearQFunction(range(n_acts), feat, params=np.zeros(k), state_dim=worlds[0].state_dim, action_dim=worlds[0].action_dim)
        r.append(np.array(transfer_rl(worlds[np.random.randint(0, len(worlds))], q, (prior_means, prior_variances), batch_size=10, epsilon=0.01, K=10, verbose=True, render=False, n_fit=10)))

    # Plot Performance
    rew = np.array(r)
    mean_r = np.mean(rew, axis=0)
    std_r = np.std(rew, axis=0)
    itr = np.arange(mean_r.size)

    plt.figure()
    plt.plot(itr, mean_r, label="Task seen before", color='blue')
    plt.fill_between(itr, mean_r + std_r, mean_r - std_r, color='blue', alpha=0.3)
    plt.ylabel("Discounted Reward")
    plt.xlabel("Iteration")
    plt.title("Transferring with Normal Prior")

    # Solve a task not seen before

    rew = list()
    for _ in range(len(worlds)):

        task = np.random.randint(0, len(worlds))
        prior_means = list()
        prior_variances = list()
        weights = [q._w for q in q_functions]
        for a in range(n_acts):
            m = sum([weights[i][a] for i in range(len(weights)) if i != task])/(len(weights)-1)
            # covar = sum(map(lambda x: x.reshape(k,1) @ x.reshape(1,k), [w[a]-m for w in weights]))/len(weights)
            covar = sum(map(lambda x: np.diag(x * x), [weights[i][a]-m for i in range(len(weights)) if i != task]))/(len(weights)-1)
            prior_means.append(m)
            prior_variances.append(covar)

        q = Q.LinearQFunction(range(n_acts), feat, params=np.zeros(k), state_dim=worlds[0].state_dim, action_dim=worlds[0].action_dim)
        r = transfer_rl(worlds[task], q, (prior_means, prior_variances), batch_size=10, epsilon=0.01, K=10, verbose=True, render=False, n_fit=10)
        rew.append(np.array(r))

    # Plot performance
    rew = np.asarray(rew)
    mean_r = np.mean(rew, axis=0)
    std_r = np.std(rew, axis= 0)
    itr = np.arange(mean_r.size)

    plt.plot(itr, mean_r, label="Task not seen before", color='red')
    plt.fill_between(itr, mean_r + std_r, mean_r-std_r, color='red', alpha=0.3)
    plt.legend()
    plt.show()