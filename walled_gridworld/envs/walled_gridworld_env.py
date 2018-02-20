import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from gym.envs.toy_text import discrete
import numpy as np

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

NORMAL_R = -1
HIT_WALL_R = -10
GOAL_R = 100


class WalledGridworld(discrete.DiscreteEnv):

    def __init__(self, n=5, passage=-1):
        self.shape = (n, n)
        self.start_state_index = np.ravel_multi_index((0, 0), self.shape)

        nS = np.prod(self.shape)
        nA = 4

        # Wall
        self._pass = np.randint(0, n) if passage == -1 else passage
        self._wall = np.zeros(self.shape, dtype=np.bool)
        self._wall[n//2, :] = True
        self._wall[n//2, self._pass] = False;

        # Calculate transition probabilities and rewards
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(nA)}
            P[s][UP] = self._calculate_transition_prob(position, [1, 0])
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            P[s][DOWN] = self._calculate_transition_prob(position, [-1, 0])
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

        # Calculate initial state distribution
        isd = np.zeros(nS)
        isd[self.start_state_index] = 1.0

        super(WalledGridworld, self).__init__(nS, nA, P, isd)

    def _limit_coordinates(self, coord):

        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord


    def _calculate_transition_prob(self, current, delta):

        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        if self._wall[tuple(new_position)]:
            return [(1.0, np.ravel_multi_index(tuple(current), self.shape), HIT_WALL_R, False)]
        terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
        is_done = tuple(new_position) == terminal_state
        return [(1.0, new_state, NORMAL_R if not is_done else GOAL_R, is_done)]

    def render(self, mode='human'):
        outfile = sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "
            # Print terminal state
            elif position == (self.shape[0]-1, self.shape[1]-1):
                output = " T "
            elif self._wall[position]:
                output = " # "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += '\n'

            outfile.write(output)
        outfile.write('\n')