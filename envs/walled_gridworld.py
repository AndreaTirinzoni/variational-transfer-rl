import gym
import numpy as np
from gym import spaces

"""
The WalledGridworld environment. The agent starts from the bottom left corner and has to
reach the upper right corner. However, there is a wall in the middle of the grid with only one
door. The position of the door might change between tasks.

Info
----
  - State space: 2D Box (x,y)
  - Action space: Discrete (UP,RIGHT,DOWN,LEFT)
  - Parameters: grid size (width,height), door location (x coordinate), door width
"""


class WalledGridworld(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    def __init__(self, size=np.array([10, 10]), door_x=5, door_width=1):
        # General MDP parameters
        self.horizon = 50
        self.gamma = 0.99
        self.state_dim = 2
        self.action_dim = 1

        # Specific MDP parameters
        self.size = size
        self.start = np.array([0, 0])
        self.goal = np.array([10, 10])
        self.goal_radius = 1
        self.noise = 0.1
        self.step_length = 1
        self.door_x = door_x
        self.door_width = door_width
        self.wall1_p1 = np.array([0, size[1] / 2])
        self.wall1_p2 = np.array([max(0, door_x - door_width), size[1] / 2])
        self.wall2_p1 = np.array([min(door_x + door_width, size[1]), size[1] / 2])
        self.wall2_p2 = np.array([size[1], size[1] / 2])

        # State space: [0,width]x[0,height]
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=size)

        # Action space: {UP,RIGHT,DOWN,LEFT}
        self.action_space = spaces.Discrete(4)

        self.reset()

    def reset(self, state=None):
        if state is None:
            self.current_position = np.array([0., 0.])
        else:
            self.current_position = np.array(state)

        return self.get_state()

    def get_state(self):
        return np.array(self.current_position)

    def step(self, a):
        s = self.get_state()
        a = int(a)
        assert 0 <= a <= 3

        # UP
        if int(a) == 0:
            self.current_position[1] += self.step_length
        # RIGHT
        elif int(a) == 1:
            self.current_position[0] += self.step_length
        # DOWN
        elif int(a) == 2:
            self.current_position[1] -= self.step_length
        # LEFT
        elif int(a) == 3:
            self.current_position[0] -= self.step_length

        # Add noise
        if self.noise > 0:
            self.current_position += np.random.normal(scale=self.noise, size=(2,))

        # Check whether the agent hit a wall
        if self._check_intersect(s,self.current_position,self.wall1_p1,self.wall1_p2) or self._check_intersect(s,self.current_position,self.wall2_p1,self.wall2_p2):
            self.current_position = s

        # Clip to make sure the agent is inside the grid
        self.current_position = self.current_position.clip([0, 0], self.size)

        # Compute reward
        if np.linalg.norm(self.current_position - self.goal) < self.goal_radius:
            absorbing = True
            reward = 0.0
        else:
            absorbing = False
            reward = -1.0

        return self.get_state(), reward, absorbing, {}

    @staticmethod
    def _check_intersect(p11, p12, p21, p22):

        d1 = p11 - p12
        d2 = p21 - p22
        n1 = np.array([d1[1], -d1[0]])
        n2 = np.array([d2[1], -d2[0]])
        b1 = -np.dot(n1, p11)
        b2 = -np.dot(n2, p21)

        A = np.array([n1, n2])
        b = np.array([b1, b2])

        if np.linalg.det(A) == 0:
            return False

        x = -np.dot(np.linalg.inv(A), b)
        in1 = min(p11[0], p12[0]) <= x[0] <= max(p11[0], p12[0]) and min(p11[1], p12[1]) <= x[1] <= max(p11[1], p12[1])
        in2 = min(p21[0], p22[0]) <= x[0] <= max(p21[0], p22[0]) and min(p21[1], p22[1]) <= x[1] <= max(p21[1], p22[1])
        return in1 and in2


