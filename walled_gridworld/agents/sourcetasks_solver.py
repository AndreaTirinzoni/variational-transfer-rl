from walled_gridworld.envs import walled_gridworld_env as wgw
import dp_solver as dp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

tasks = []
q_stars = []

N = 5

for i in range(N):
    tasks.append(wgw.WalledGridworld(N, i))

for t in tasks:
    q_stars.append(dp.qvalue_iteration(t))


qall = np.asanyarray(q_stars)   # Optimal Q functions in array form
qavg = np.average(qall, 0)      # Average optimal Q function
qmin = np.min(qall, 0)
qmax = np.max(qall, 0)


start = (0, 0)
close_goal = (N-2, N-2)
wall1 = (N//2, N//2)
wall2 = (N//2 + 1, N//2)
wall3 = (N//2 - 1, N//2)

actions = ("UP", "RIGHT", "DOWN", "LEFT")

for a in range(4):
    fig, axs = plt.subplots(N, N)
    fig.suptitle("Action: " + actions[a])
    for j in range(N):
        for i in range(N):
            axs[i, j].hist(qall[:, np.ravel_multi_index((i, j), (N, N)), a], bins='doane')
            axs[i, j].set_title("Position " + str((i, j)))



# State Space

X = np.arange(N)
Y = np.arange(N)
X, Y = np.meshgrid(X, Y)
Vavg = np.average(np.max(qall, 2), 0)
Vstd = np.std(np.max(qall, 2), 0)
VavgGrid = Vavg.reshape((N, N))

f = plt.figure()
ax = f.gca(projection='3d')
ax.plot_surface(X, Y, VavgGrid)
ax.plot_surface(X, Y, (Vavg + Vstd).reshape((N, N)))
ax.plot_surface(X, Y, (Vavg - Vstd).reshape((N, N)))
ax.set_title("Value Function")
plt.show()