import numpy as np
from algorithms.simple_rl import simple_RL
import features.grbf as grbf
import envs.marcellos_gridworld as env
import algorithms.linear_q_function as Q
import matplotlib.pyplot as plt


if __name__ == '__main__':
    n = 8
    acts = 4
    k = (n + 1) * (n + 1)

    world = env.MarcellosGridworld(np.array((n, n)), door_x=(n-.5, 0.5))
    mean = np.array([[x, y] for x in range(0, n + 1) for y in range(0, n + 1)])
    variance = (np.ones(k) / 3) ** 2
    features = grbf.GaussianRBF(mean, variance, K=k, dims=world.state_dim)
    q = Q.LinearQFunction(range(acts), features, np.zeros(k), state_dim=world.state_dim, action_dim=world.action_dim)
    episodes = 50

    # plt.ion()
    plt.gca()
    r = simple_RL(world, q, epsilon=0.2, K=episodes, render=True, verbose=True, batch_size=10)