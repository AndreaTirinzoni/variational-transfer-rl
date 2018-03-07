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
        self.goal = size
        self.goal_radius = 1
        self.noise = 0.2
        self.step_length = 1
        self.door_x = door_x
        self.door_width = door_width
        self.wall1_p1 = np.array([0, size[1] / 2])
        self.wall1_p2 = np.array([max(0., door_x - door_width / 2), size[1] / 2])
        self.wall2_p1 = np.array([min(door_x + door_width / 2, size[1]), size[1] / 2])
        self.wall2_p2 = np.array([size[1], size[1] / 2])

        # State space: [0,width]x[0,height]
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=size)

        # Action space: {UP,RIGHT,DOWN,LEFT}
        self.action_space = spaces.Discrete(4)

        self.reset()
        self.viewer = None

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
            # print("UP")
            self.current_position[1] += self.step_length
        # RIGHT
        elif int(a) == 1:
            # print("RIGHT")
            self.current_position[0] += self.step_length
        # DOWN
        elif int(a) == 2:
            # print("DOWN")
            self.current_position[1] -= self.step_length
        # LEFT
        elif int(a) == 3:
            # print("LEFT")
            self.current_position[0] -= self.step_length

        # Add noise
        if self.noise > 0:
            self.current_position += np.random.normal(scale=self.noise, size=(2,))

        # Clip to make sure the agent is inside the grid
        self.current_position = self.current_position.clip([0, 0], self.size)

        # Check whether the agent hit a wall
        if self._check_intersect(s, self.current_position, self.wall1_p1, self.wall1_p2) or \
                self._check_intersect(s, self.current_position, self.wall2_p1, self.wall2_p2):
            self.current_position = s

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
        in1 = min(p11[0], p12[0]) - 1e-5 <= x[0] <= max(p11[0], p12[0]) + 1e-5 and \
              min(p11[1], p12[1]) - 1e-5 <= x[1] <= max(p11[1], p12[1]) + 1e-5
        in2 = min(p21[0], p22[0]) - 1e-5 <= x[0] <= max(p21[0], p22[0]) + 1e-5 and \
              min(p21[1], p22[1]) - 1e-5 <= x[1] <= max(p21[1], p22[1]) + 1e-5
        return in1 and in2

    def _render(self, mode='human', close=False, a=None):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(0, self.size[0], 0, self.size[1])

        if self.current_position is None:
            return None

        self.viewer.draw_line(self.wall1_p1, self.wall1_p2)
        self.viewer.draw_line(self.wall2_p1, self.wall2_p2)
        c = self.viewer.draw_circle(radius=0.2)
        c.set_color(0, 0, 0)
        if a == 0:
            c.set_color(0.8, 0.8, 0)
        c.add_attr(rendering.Transform(translation=(1, self.size[1] - 0.5)))
        c = self.viewer.draw_circle(radius=0.2)
        c.set_color(0, 0, 0)
        if a == 1:
            c.set_color(0.8, 0.8, 0)
        c.add_attr(rendering.Transform(translation=(1.5, self.size[1] - 1)))
        c = self.viewer.draw_circle(radius=0.2)
        c.set_color(0, 0, 0)
        if a == 2:
            c.set_color(0.8, 0.8, 0)
        c.add_attr(rendering.Transform(translation=(1, self.size[1] - 1.5)))
        c = self.viewer.draw_circle(radius=0.2)
        c.set_color(0, 0, 0)
        if a == 3:
            c.set_color(0.8, 0.8, 0)
        c.add_attr(rendering.Transform(translation=(0.5, self.size[1] - 1)))

        goal = self.viewer.draw_circle(radius=self.goal_radius)
        goal.set_color(0, 0.8, 0)
        goal.add_attr(rendering.Transform(translation=(self.goal[0], self.goal[1])))

        agent = self.viewer.draw_circle(radius=0.1)
        agent.set_color(.8, 0, 0)
        transform = rendering.Transform(translation=(self.current_position[0], self.current_position[1]))
        agent.add_attr(transform)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
