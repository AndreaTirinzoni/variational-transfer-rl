import gym
import numpy as np
from gym import spaces
import time

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
        self.horizon = 100
        self.gamma = 0.99
        self.state_dim = 22
        self.action_dim = 1
        self.time_step = 1
        self.speed = 0.5
        self.angular_speed = np.pi/4
        self.wall_dim = wall_dim
        self.range = 2.

        # Specific MDP parameters
        self.size = size
        self.start = start_pos
        self.goal = size if goal_pos is None else goal_pos
        self.goal_radius = 1
        self.noise = 0.05

        # State space: [0,width,0]x[0,height,2pi]
        self.observation_space = spaces.Box(low=np.array([0., 0., 0.]), high=np.concatenate((size, np.array([2*np.pi]))))

        # Load walls
        assert walls.shape == tuple(np.floor(self.size / self.wall_dim))
        self.walls = walls
        # Action space: {FORWARD,ROTATE_LEFT,ROTATE_RIGHT}
        self.action_space = spaces.Discrete(3)

        self.goal_tile = tuple(self.get_tiled_state(self.goal).astype("int"))
        self.reset()
        self.viewer = None

    def reset(self, state=None):
        if state is None:
            self.walls[self.goal_tile] = -1.
            self.current_state = np.array(np.concatenate((self.start, np.zeros(1))))
            free_cells = np.array(np.where(self.walls == 0))
            choice = np.random.choice(free_cells.shape[1])
            self.current_state = np.array((free_cells[0, choice] + np.random.ranf(), free_cells[1, choice] + np.random.ranf(), 0.))
            self.walls[self.goal_tile] = 0.
        else:
            if state.size == 22:
                self.current_state = np.array((state[0], state[1], np.arctan2(state[3],state[2])))
            elif state.size == 3:
                self.current_state = np.array(state)
        return self.get_observation()

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
        self.current_state = self._check_intersect(s, self.current_state)

        # Compute reward
        if self._is_goal(self.get_tiled_state(self.current_state[:2])):
            absorbing = True
            reward = 1.0
        else:
            absorbing = False
            reward = 0.0

        return self.get_observation(), reward, absorbing, {}

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
        return np.floor((s-1e-8)/self.wall_dim)

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
        return np.array_equal(tile, np.floor((self.goal-1e-8)/self.wall_dim))


    # TODO : maybe recast with _intersection_tile function
    def _check_intersect(self, s1, s2):
        # compute line
        d = s2-s1
        vertical = False
        c = 1e-10

        if np.abs(d[0]) == 0.0 and np.abs(d[1]) == 0.0:
            return s2

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
                        return np.array((h, m*h + b, s2[2]))
                    else:
                        x += dir[0]
            elif y != tile2[1] and Maze._check_tile_intersect(m, b, x, x+1, y+dir[1], y+dir[1]+1, vertical) :
                if self.walls[x, y+dir[1]] == 1:
                    v = y + c if dir[1] < 0 else y + 1 - c
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

        if np.abs(d[0]) >= c:
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
        c = 1e-8
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
        return x + direction * np.min(a) - c

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
        print(maze[4][3])
        for i in range(1000):
            a = np.random.randint(0, 3)
            s, _, _, _ = m.step(a)
            print("Iter {} State {} A {}".format(i,s,a))
            print(m.get_observation())
            # m._render(a=a)
            time.sleep(.1)