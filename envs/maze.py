import gym
import numpy as np
from gym import spaces
import time

"""
Info
----
  - State space: 3D Box (x,y,theta)
  - Action space: Discrete (FORWARD, BACKWARD, ROTATE_LEFT, ROTATE_RIGHT)
  - Parameters: grid size (width,height), wall_dim (width, height), goal_pos, start_pos, walls: np.ndarray
"""


class Maze(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    def __init__(self, size=np.array([10., 10.]), wall_dim=np.array((1.,1.)), start_pos=np.array((0,0)), goal_pos=None, walls=None):
        # General MDP parameters
        self.horizon = 50
        self.gamma = 0.99
        self.state_dim = 3
        self.action_dim = 1
        self.time_step = 1
        self.speed = 1.5
        self.angular_speed = np.pi/4
        self.wall_dim = wall_dim

        # Specific MDP parameters
        self.size = size
        self.start = start_pos
        self.goal = size if goal_pos is None else goal_pos
        self.goal_radius = 1
        self.noise = 0.2

        # State space: [0,width,0]x[0,height,2pi]
        self.observation_space = spaces.Box(low=np.array([0., 0., 0.]), high=np.concatenate((size, np.array([2*np.pi]))))

        # Load walls
        assert walls.shape == tuple(np.floor(self.size / self.wall_dim))
        self.walls = walls

        # Action space: {FORWARD,BACKWARD,ROTATE_LEFT,ROTATE_RIGHT}
        self.action_space = spaces.Discrete(4)

        self.reset()
        self.viewer = None

    def reset(self, state=None):
        if state is None:
            self.current_state = np.array(np.concatenate((self.start, np.zeros(1))))
        else:
            self.current_state = np.concatenate((np.array(state), np.zeros(1)))
        return self.get_state()

    def get_state(self):
        return np.array(self.current_state)

    def step(self, a):
        s = self.get_state()
        a = int(a)
        assert 0 <= a <= 3

        # FORWARD
        if int(a) == 0:
            self.current_state[:2] += self.speed * self.time_step \
                                      * np.array((np.cos(self.current_state[-1]), np.sin(self.current_state[-1])))
        # ROTATE LEFT
        elif int(a) == 1:
            self.current_state[-1] += self.angular_speed * self.time_step
        # ROTATE RIGHT
        elif int(a) == 2:
            self.current_state[-1] -= self.angular_speed * self.time_step

        # Add noise
        if self.noise > 0:
            self.current_state += np.random.normal(scale=self.noise, size=(3,))

        # Clip to make sure the agent is inside the grid
        self.current_state[-1] = np.divmod(self.current_state[-1], 2 * np.pi)[1]    # set the orientation to be [0,2pi]

        clipped = self.current_state[:2].clip([0, 0], self.size - 1e-5)
        self.current_state[:2] = clipped

        # Check whether the agent hit a wall
        self.current_state = self._check_intersect(s, self.current_state)

        # Compute reward
        if np.linalg.norm(self.current_state[:2] - self.goal) < self.goal_radius:
            absorbing = True
            reward = 1.0
        else:
            absorbing = False
            reward = 0.0

        return self.get_state(), reward, absorbing, {}

    def _check_intersect(self, s1, s2):
        # compute line
        d = s2-s1
        vertical = False
        c = 1e-5

        if d[0] == 0. and d[1] == 0.:
            return s2

        if d[0] != 0.:
            m = d[1]/d[0]
            b = s1[1]-m*s1[0]
        else:
            m = 0.
            b = s1[0]
            vertical = True

        tile1 = np.floor(s1[:2]/self.wall_dim).astype('int')
        tile2 = np.floor(s2[:2]/self.wall_dim).astype('int')
        x,y = tuple(tile1)
        dir = np.sign(tile2-tile1).astype('int')
        while x != tile2[0] or y != tile2[1]:
            if x != tile2[0] and not vertical and Maze._check_tile_intersect(m, b, x+dir[0], x+dir[0]+1, y, y+1):
                    if self.walls[x+dir[0], y] == 1:
                        h = x + c if dir[0] < 0 else x + 1 - c
                        return np.array((h, m*h + b, s2[2]))
                    else:
                        x += dir[0]
            elif y != tile2[1] and Maze._check_tile_intersect(m, b, x, x+1, y+dir[1], y+dir[1]+1, vertical) :
                if self.walls[x, y+dir[1]] == 1:
                    v = y + c if dir[0] < 0 else y + 1 - c
                    return np.array(((v-b)/m, v, s2[2])) if not vertical else np.array((x, v, s2[2]))
                else:
                    y += dir[1]
            elif Maze._check_tile_intersect(m, b, x+dir[0], x+dir[0]+1, y+dir[1], y+dir[1]+1, vertical):
                    if self.walls[x+dir[0], y+dir[1]] == 1:
                        h = x + c if dir[0] < 0 else x + 1 - c
                        v = y + c if dir[0] < 0 else y + 1 - c
                        return np.array((h,v,s2[2]))
                    else:
                        x += dir[0]
                        y += dir[1]
        return s2   # go to final state

    @staticmethod
    def _check_tile_intersect(m, b, x1, x2, y1, y2, vertical=False):

        if m == 0. and vertical:
            return b >= x1 and b < x2

        if m == 0. and not vertical:
            return b >= y1 and b < y2

        t1 = m * x1 + b
        t2 = m * x2 + b
        t3 = (y1 - b) / m
        t4 = (y2 - b) / m

        b1 = t1 < y2 and t1 >= y1
        b2 = t2 < y2 and t2 >= y1
        b3 = t3 >= x1 and t3 < x2
        b4 = t4 >= x1 and t4 < x2

        return b1 or b2 or b3 or b4

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

        if self.current_state is None:
            return None

        # draw obstacles
        x = np.arange(0, self.size[0]+1, self.wall_dim[0])
        y = np.arange(0, self.size[1]+1, self.wall_dim[1])
        for i in range(x.shape[0]-1):
            for j in range(y.shape[0]-1):
                if self.walls[i,j] == 1:
                    vertices = [(x[k], y[l]) for k in range(i,i+2) for l in range(j, j+2)]
                    self.viewer.draw_polygon(vertices)

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
        orientation = self.viewer.draw_line([0.,0.], [.1 * np.cos(self.current_state[2]), .1 * np.sin(self.current_state[2])])
        agent.set_color(.8, 0, 0)
        transform = rendering.Transform(translation=(self.current_state[0], self.current_state[1]))
        agent.add_attr(transform)
        orientation.add_attr(transform)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

if __name__ == '__main__':
    import utils
    mazes = utils.load_object("../scripts/mazes10x10")
    for maze in mazes:
        m = Maze(size=maze[0], wall_dim=maze[1], goal_pos=maze[2], start_pos=maze[3], walls=maze[4])
        print(maze[4][3])
        for i in range(100):
            a = np.random.randint(0, 3)
            s, _, _, _ = m.step(a)
            print("State {} A {}".format(s,a))
            m._render(a=a)
            time.sleep(1)