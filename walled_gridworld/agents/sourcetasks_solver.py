from walled_gridworld.envs import walled_gridworld_env as wgw
import dp_solver as dp

tasks = []
q_stars = []

for i in range(5):
    tasks.append(wgw.WalledGridworld(5, i))

for t in tasks:
    q_stars.append(dp.qvalue_iteration(t))

print(q_stars)

