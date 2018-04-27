import gym
import numpy as np
from gym import spaces
import time
import matplotlib.pyplot as plt

"""
Info
----
  - State space: 3D Box (x,y,theta). Observation: (x,y,z,cos(theta),sin(theta), obstacles[-pi/2 + theta, ... , +pi/2+theta], goal[-pi/2 + theta, +pi/2 + theta])
  - Action space: Discrete (FORWARD, ROTATE_LEFT, ROTATE_RIGHT)
  - Parameters: grid size (width,height), wall_dim (width, height), goal_pos, start_pos, walls: np.ndarray
"""


class Maze(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    def __init__(self, size=np.array([10., 10.]), wall_dim=np.array((1.,1.)), start_pos=np.array((0., 0.)), goal_pos=None, walls=None):
        # General MDP parameters
        self.horizon = 1
        self.gamma = 0.99
        self.state_dim = 22
        self.absolute_state_dim = 3
        self.action_dim = 1
        self.time_step = 1
        self.speed = 0.5
        self.angular_speed = np.pi/8
        self.wall_dim = wall_dim
        self.range = 2.

        # Specific MDP parameters
        self.size = size
        self.start = start_pos
        self.goal = size if goal_pos is None else goal_pos
        self.goal_radius = 1
        self.noise = 0.05
        self.irew = False  # True if informative reward is to be use
        self.binarized = False # True if observation is to be returned binarized

        # State space: [0,width,0]x[0,height,2pi]
        self.observation_space = spaces.Box(low=np.array([0., 0., 0.]), high=np.concatenate((size, np.array([2*np.pi]))))

        # Load walls
        assert walls.shape == tuple(np.floor(self.size / self.wall_dim))
        self.walls = walls
        # Action space: {FORWARD,ROTATE_LEFT,ROTATE_RIGHT}
        self.action_space = spaces.Discrete(3)

        self.goal_tile = tuple(self.get_tiled_state(self.goal-1e-8).astype("int"))
        self.reset()
        self.viewer = None

    def reset(self, state=None):
        if state is None:
            self.walls[self.goal_tile] = -1.
            free_cells = np.array(np.where(self.walls == 0))
            choice = np.random.choice(free_cells.shape[1])
            self.current_state = np.array((free_cells[0, choice] + np.random.ranf(), \
                                           free_cells[1, choice] + np.random.ranf(), \
                                           np.random.ranf() * 2 * np.pi))
            self.walls[self.goal_tile] = 0.
        else:
            if state.size == self.state_dim:
                self.current_state = np.array((state[0], state[1], np.arctan2(state[3],state[2])))
            elif state.size == self.absolute_state_dim:
                assert state[0] < self.size[0] and state[0] >= 0 and \
                       state[1] < self.size[1] and state[1] >= 0
                self.current_state = np.array((state[0], state[1], np.divmod(state[2], 2*np.pi)[1]))
        return self.get_observation() if not self.binarized else self.get_binarized_observation()

    def get_state(self):
        return np.array(self.current_state)

    def step(self, a):
        s = self.get_state()
        a = int(a)
        assert 0 <= a <= 2

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

        clipped = self.current_state[:2].clip([0., 0.], self.size - 1e-8)
        self.current_state[:2] = clipped

        # Check whether the agent hit a wall
        self.current_state, hits = self._check_intersect(s, self.current_state)

        # Compute reward
        reward = 0.
        absorbing = False
        if self.irew & hits:   # if it hits a wall
            reward -= 0.1
        if self.irew & a > 0: # if does not move forward
            reward -= 0.001
        if self._is_goal(self.get_tiled_state(self.current_state[:2])):
            absorbing = True
            reward += 1.0

        return self.get_observation() if not self.binarized else self.get_binarized_observation(), reward, absorbing, {}

    def get_observation(self):
        s = self.get_state()
        phi = np.linspace(-np.pi/2, np.pi/2, 9) + s[2]
        d = (s[:2, np.newaxis] + self.range * np.array([np.cos(phi), np.sin(phi)])).T
        obstacles = []
        goals = []
        for i in range(phi.size):
            dist, goal = self._get_obstacle_goal(s[:2],d[i])
            obstacles.append(np.linalg.norm(dist-s[:2]))
            goals.append(1. if goal else 0.)
        return np.concatenate((s[:2], np.array([np.cos(s[2]), np.sin(s[2])]), np.array(obstacles), np.array(goals)))

    def get_tiled_state(self, s):
        return np.floor(s/self.wall_dim).astype("int")

    def get_binarized_observation(self, observation=None):
        """ Takes a observation and binarizes the absolute position based on the tiled partition given by the
        matrix self.walls. It puts a 1. in the tile currently occupied by the agent. It returns it in the first
        positions of the new binarized observation. If no observation is passed, the current observation is used. """

        if observation is None:
            observation = self.get_observation()
        if observation.ndim == 1:
            occupied_tile = self.get_tiled_state(observation[:2])
            tiles = np.zeros(self.walls.shape)
            tiles[occupied_tile[0], occupied_tile[1]] = 1.
            return np.concatenate((tiles.flatten(), observation[2:]))
        if observation.ndim == 2:
            occupied_tile = np.floor(observation[:, :2]/self.wall_dim[np.newaxis]).astype("int")
            tiles = np.zeros((observation.shape[0], self.walls.shape[0], self.walls.shape[1]))
            s = np.arange(observation.shape[0])
            tiles[s, occupied_tile[s,0], occupied_tile[s,1]] = 1.
            return np.concatenate((tiles.reshape(observation.shape[0], self.walls.shape[0]*self.walls.shape[1]), observation[:,2:]), axis=1)

    def _get_obstacle_goal(self, s1, s2):
        tiles = self._get_traversed_tiles(s1, s2)
        goal = False
        for t in tiles:
            if t[0] <= self.size[0]-1 and t[0] >= 0 and t[1] <= self.size[1]-1 and t[1] >= 0: #inside the grid
                if self.walls[t[0],t[1]] == 1:
                    return Maze._intersection_tile(s1, s2-s1, t), self._is_goal(t)
                goal = self._is_goal(t)
            else:   # first tile outside the grid
                return Maze._intersection_tile(s1, s2-s1, t), goal
        return s2, goal

    def _is_goal(self, tile):
        return np.array_equal(tile, self.goal_tile)

    def toggle_irew(self):
        self.irew = not self.irew

    def toggle_binarized(self):
        self.binarized = not self.binarized
        if self.binarized:
            self.state_dim = self.walls.size + 20
        else:
            self.state_dim = 22

    # TODO : maybe recast with _intersection_tile function
    def _check_intersect(self, s1, s2):

        """ Return the new position in the direction of the displacement that can be reached. Also, returns whether it
        hit a wall."""
        # compute line
        d = s2-s1
        vertical = False
        c = 1e-10

        if np.abs(d[0]) == 0.0 and np.abs(d[1]) == 0.0:
            return s2, False

        if np.abs(d[0]) >= c:
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
                        return np.array((h, m*h + b, s2[2])), True
                    else:
                        x += dir[0]
            elif y != tile2[1] and Maze._check_tile_intersect(m, b, x, x+1, y+dir[1], y+dir[1]+1, vertical) :
                if self.walls[x, y+dir[1]] == 1:
                    v = y + c if dir[1] < 0 else y + 1 - c
                    return np.array(((v-b)/m, v, s2[2])) if not vertical else np.array((x, v, s2[2])), True
                else:
                    y += dir[1]
            elif Maze._check_tile_intersect(m, b, x+dir[0], x+dir[0]+1, y+dir[1], y+dir[1]+1, vertical):
                    if self.walls[x+dir[0], y+dir[1]] == 1:
                        h = x + c if dir[0] < 0 else x + 1 - c
                        v = y + c if dir[0] < 0 else y + 1 - c
                        return np.array((h,v,s2[2])), True
                    else:
                        x += dir[0]
                        y += dir[1]
        return s2, False   # go to final state

    # TODO : maybe recast with _intersection_tile function
    def _get_traversed_tiles(self, s1, s2):
        """ returns a list of tiles traversed by the ray define by its extreme points s1, s2 """
        # compute line
        d = s2 - s1
        vertical = False
        c = 1e-8

        tile1 = np.floor(s1[:2] / self.wall_dim).astype('int')
        tile2 = np.floor(s2[:2] / self.wall_dim).astype('int')
        x, y = tuple(tile1)
        dir = np.sign(tile2 - tile1).astype('int')
        traversed = [tile1,]

        if np.abs(d[0]) > 0:
            m = d[1] / d[0]
            b = s1[1] - m * s1[0]
        else:
            m = 0.
            b = s1[0]
            vertical = True

        while x != tile2[0] or y != tile2[1]:
            if x != tile2[0] and not vertical and Maze._check_tile_intersect(m, b, x + dir[0], x + dir[0] + 1, y, y + 1):
                x += dir[0]
                traversed.append(np.array((x,y)))
            elif y != tile2[1] and Maze._check_tile_intersect(m, b, x, x + 1, y + dir[1], y + dir[1] + 1, vertical):
                y += dir[1]
                traversed.append(np.array((x, y)))
            elif Maze._check_tile_intersect(m, b, x + dir[0], x + dir[0] + 1, y + dir[1], y + dir[1] + 1, vertical):
                x += dir[0]
                y += dir[1]
                traversed.append(np.array((x, y)))
        return traversed

    @staticmethod
    def _intersection_tile(x, direction, tile):
        """
        Computes the intersection point with the given tile (np.array) with the ray starting at x(np.array)
        in the given direction(np.array) (only forward)
        """
        c = 1e-8 # tolerance
        borders = np.array((tile, tile + 1)).T
        if np.abs(direction[0]) <= c:
            a = (borders[1] - x[1]) / direction[1]  # compute intersections
            a = a[a >= 0]
        elif np.abs(direction[1]) <= c:
            a = (borders[0] - x[0]) / direction[0]
            a = a[a >= 0]
        else:
            a = (borders - x[:, np.newaxis]) / direction[:, np.newaxis]
            a = a.flatten()
            a = a[a >= 0]

        a = np.sort(a)  # sort to have the closest intersections first
        cuts = x[:, np.newaxis] + direction[:, np.newaxis] * a[np.newaxis]                     # compute intersections
        idx = np.all(np.logical_and(np.less_equal(cuts-c, borders[:,1][:,np.newaxis]),\
                                      np.less_equal(borders[:,0][:,np.newaxis],cuts + c)), axis=0) # in the tile
        idx = np.where(idx)[0] # take closest intersection in the tile
        cuts = cuts[:, idx[0]]

        return cuts

    # TODO : maybe recast with _intersection_tile function
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
                    s = self.viewer.draw_polygon(v=[(0,0),(0,1),(1,1),(1,0)])
                    s.add_attr(rendering.Transform(translation=(i,j)))
                    s.set_color(0, 0, 0)

        goal = self.viewer.draw_polygon(v=[(0,0),(0,1),(1,1),(1,0)])
        goal.add_attr(rendering.Transform(translation=(self.goal_tile[0], self.goal_tile[1])))
        goal.set_color(0, 0.8, 0)

        rect = self.viewer.draw_polygon(v=[(-0.1,-0.1),(-0.1,0.1),(0.3,0.1),(0.3,-0.1)])
        rect.set_color(0, 0, .8)
        transform = rendering.Transform(translation=(self.current_state[0], self.current_state[1]),rotation=self.current_state[2])
        rect.add_attr(transform)

        agent = self.viewer.draw_circle(radius=0.15)
        agent.set_color(.8, 0, 0)
        transform = rendering.Transform(translation=(self.current_state[0], self.current_state[1]))
        agent.add_attr(transform)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

if __name__ == '__main__':
    import utils
    mazes = utils.load_object("../scripts/mazes10x10")
    for maze in mazes:
        m = Maze(size=maze[0], wall_dim=maze[1], goal_pos=maze[2], start_pos=maze[3], walls=maze[4])

        m.reset(np.array((0., 0., 0.)))
        s = m.get_observation()
        print(s)
        phi = np.linspace(-np.pi/2, np.pi/2, 9) # view angles
        o = np.linspace(0., 1., 10)
        r = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])  # rotation matrices

        sq = np.array([(0,0), (0,1), (1,0), (1,1)])
        w = np.array(np.where(maze[4] == 1)).T
        w = np.concatenate((w, np.array([m.goal_tile])), axis=0)
        sq = sq[:, np.newaxis] + w[np.newaxis]

        obstacles = s[4:13, np.newaxis] * (s[2:4])[np.newaxis]
        orientation = o[:, np.newaxis] * (s[2:4])[np.newaxis] + s[:2][np.newaxis]
        p = np.einsum("mnp,pn->pm", r, obstacles) + s[:2][np.newaxis]

        plt.xlim((-0.5, 10.5))
        plt.ylim((-0.5, 10.5))
        plt.plot(p[:, 0], p[:, 1], "kD", s[0], s[1], "ro", orientation[:, 0], orientation[:, 1], "r-")
        for k in range(sq.shape[1]-1):
            plt.plot(sq[(0, 1), k, 0], sq[(0, 1), k, 1], "k-")
            plt.plot(sq[(0, 2), k, 0], sq[(0, 2), k, 1], "k-")
            plt.plot(sq[(1, 3), k, 0], sq[(1, 3), k, 1], "k-")
            plt.plot(sq[(2, 3), k, 0], sq[(2, 3), k, 1], "k-")

        plt.plot(sq[(0, 1), -1, 0], sq[(0, 1), -1, 1], "g-")
        plt.plot(sq[(0, 2), -1, 0], sq[(0, 2), -1, 1], "g-")
        plt.plot(sq[(1, 3), -1, 0], sq[(1, 3), -1, 1], "g-")
        plt.plot(sq[(2, 3), -1, 0], sq[(2, 3), -1, 1], "g-")

        plt.show()

        for i in range(1000):
            a = np.random.randint(0, 3)
            s, _, _, _ = m.step(a)

            obstacles = s[4:13, np.newaxis] * (s[2:4])[np.newaxis]
            orientation = o[:, np.newaxis] * (s[2:4])[np.newaxis] + s[:2][np.newaxis]
            p = np.einsum("mnp,pn->pm", r, obstacles) + s[:2][np.newaxis]

            plt.xlim((-0.5, 10.5))
            plt.ylim((-0.5, 10.5))
            plt.plot(p[:, 0], p[:, 1], "kD", s[0], s[1], "ro", orientation[:, 0], orientation[:, 1], "r-")
            for k in range(sq.shape[1] - 1):
                plt.plot(sq[(0, 1), k, 0], sq[(0, 1), k, 1], "k-")
                plt.plot(sq[(0, 2), k, 0], sq[(0, 2), k, 1], "k-")
                plt.plot(sq[(1, 3), k, 0], sq[(1, 3), k, 1], "k-")
                plt.plot(sq[(2, 3), k, 0], sq[(2, 3), k, 1], "k-")

            plt.plot(sq[(0, 1), -1, 0], sq[(0, 1), -1, 1], "g-")
            plt.plot(sq[(0, 2), -1, 0], sq[(0, 2), -1, 1], "g-")
            plt.plot(sq[(1, 3), -1, 0], sq[(1, 3), -1, 1], "g-")
            plt.plot(sq[(2, 3), -1, 0], sq[(2, 3), -1, 1], "g-")

            plt.show()

            print("Iter {} State {} A {}".format(i,s,a))
            print(m.get_binarized_observation())
            # m._render(a=a)
            time.sleep(1)