import numpy as np
import envs.walled_gridworld as env
import utils
import algorithms.e_greedy_policy as policy
from algorithms.nn_q_function import NNQ
import matplotlib.pyplot as plt


def plot_Q(Q):

    V = [[], [], [], []]
    X = np.arange(0.0, 5.1, 0.1)
    for x in X:
        for y in X:
            vals = Q.compute_all_actions(np.array([x, y]))
            V[0].append(vals[0])
            V[1].append(vals[1])
            V[2].append(vals[2])
            V[3].append(vals[3])
    V = [np.flip(np.array(v).reshape(X.size, X.size), axis=0) for v in V]
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(V[0], cmap="hot", interpolation="gaussian")
    ax[0, 0].set_title("UP")
    ax[0, 1].imshow(V[1], cmap="hot", interpolation="gaussian")
    ax[0, 1].set_title("RIGHT")
    ax[1, 0].imshow(V[2], cmap="hot", interpolation="gaussian")
    ax[1, 0].set_title("DOWN")
    ax[1, 1].imshow(V[3], cmap="hot", interpolation="gaussian")
    ax[1, 1].set_title("LEFT")
    plt.show()


def solve_task(mdp, Q, epsilon=0.2, n_iter=20, n_fit=1, batch_size=1, render=False, verbose=False):
    pi_exp = policy.eGreedyPolicy(Q, Q.actions, epsilon)
    pi_eval = policy.eGreedyPolicy(Q, Q.actions, 0)

    rewards = []
    rewards.append(utils.evaluate_policy(mdp, pi_eval, initial_states=[np.array([0., 0.]) for _ in range(5)], render=render)[0])
    if verbose:
        print("===============================================")
        print("Zero-shot: {0}".format(rewards[-1]))
        print("===============================================")

    dataset = []

    for i in range(n_iter):
        dataset.append(utils.generate_episodes(mdp, pi_exp, batch_size))
        samples = np.concatenate(dataset)
        t, s, a, r, s_prime, absorbing, sa = utils.split_data(samples, 2, 1)

        for k in range(n_fit):
            y = Q.compute_bellman_target(r, s_prime, absorbing)
            Q.fit(s, a, y)

        if verbose:
            plot_Q(Q)

        rewards.append(utils.evaluate_policy(mdp, pi_eval, initial_states=[np.array([0., 0.]) for _ in range(5)],render=render)[0])
        if verbose:
            print("===============================================")
            print("Iteration " + str(i))
            print("Reward: " + str(rewards[-1]))
            print("===============================================")
    return rewards


size = 5
n_actions = 4
mdp = env.WalledGridworld(np.array((size, size)), door_x=2.5)
Q = NNQ(range(n_actions), 2, gamma=mdp.gamma)
n_iter = 20
batch_size = 10
n_fit = 1
epsilon = 0.2
render = False
verbose = True

rewards = solve_task(mdp,Q,epsilon,n_iter,n_fit,batch_size,render,verbose)
print(rewards)
rewards = np.array(rewards)
iters = np.arange(rewards.shape[0])
plt.plot(iters,rewards)
plt.show()


