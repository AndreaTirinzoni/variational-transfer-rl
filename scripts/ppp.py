import numpy as np
import matplotlib.pyplot as plt
import utils

nt_states = np.array(utils.load_object("nt_states"))
gvt_states = np.array(utils.load_object("gvt_states"))
mgvt_states = np.array(utils.load_object("mgvt_states"))

colors = ['#4EACC5', '#FF9C34', '#4E9A06']
plt.figure()
plt.hold(True)
plt.plot(nt_states[:,0], nt_states[:,1], 'w', markerfacecolor=colors[0], marker='.', markersize=6, label="NT")
plt.plot(gvt_states[:,0], gvt_states[:,1], 'w', markerfacecolor=colors[1], marker='.', markersize=6, label="GVT")
plt.plot(mgvt_states[:,0], mgvt_states[:,1], 'w', markerfacecolor=colors[2], marker='.', markersize=6, label="1-MGVT")
plt.plot([0,4.5],[5,5], color="#000000", linewidth=3)
plt.plot([5.5,10],[5,5], color="#000000", linewidth=3)
plt.plot([0,10],[0,0], color="#000000", linewidth=3)
plt.plot([0,10],[10,10], color="#000000", linewidth=3)
plt.plot([0,0],[0,10], color="#000000", linewidth=3)
plt.plot([10,10],[0,10], color="#000000", linewidth=3)

plt.legend(loc='best', numpoints=1, fancybox=True, frameon=False, markerscale=3)

plt.grid(True)
plt.savefig("exploration.png", format='png')
plt.show()



nt_transitions = utils.load_object("nt_transitions")
gvt_transitions = utils.load_object("gvt_transitions")

colors = ['#4EACC5', '#FF9C34', '#4E9A06']
plt.figure()
plt.hold(True)

for i in range(len(nt_transitions)):
    x = np.array(nt_transitions[i])
    plt.plot(x[:,0], x[:,1], color=colors[0])

for i in range(len(gvt_transitions)):
    x = np.array(gvt_transitions[i])
    plt.plot(x[:,0], x[:,1], color=colors[1])

plt.plot([0,4.5],[5,5], color="#000000", linewidth=3)
plt.plot([5.5,10],[5,5], color="#000000", linewidth=3)
plt.plot([0,10],[0,0], color="#000000", linewidth=3)
plt.plot([0,10],[10,10], color="#000000", linewidth=3)
plt.plot([0,0],[0,10], color="#000000", linewidth=3)
plt.plot([10,10],[0,10], color="#000000", linewidth=3)

plt.legend(loc='best', numpoints=1, fancybox=True, frameon=False, markerscale=3)

plt.grid(True)
plt.savefig("exploration2.png", format='png')
plt.show()
