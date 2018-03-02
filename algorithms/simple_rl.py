import numpy as np
import envs.walled_gridworld as env
import utils
import algorithms.e_greedy_policy as policy
import algorithms.linear_q_function as Q

"""
Implementation of a simple RL algorithm based on an e-greedy policy and linearly parameterized Q function
"""
def simple_RL(mpd, Q, epsilon=0):

    pol = policy.eGreedyPolicy(Q, mpd.action_space, epsilon)
    samples = utils.generate_episodes(mpd, pol, 100)