import numpy as np

"""
Apply iteratively the Optimal Bellman Operator to solve the Environment.
"""
def qvalue_iteration(env, gamma=0.99, theta=0.01):
    q = np.zeros((env.nS, env.nA))
    change = theta+1
    while (abs(change) > theta):
        deltas = []
        for s in range(0, env.nS):
            for a in range(0, env.nA):
                delta = -q[s, a] + (env.P[s][a][0][2] + gamma*expected_max_q_value(s, a, q, env))
                q[s, a] = q[s, a] + delta
                deltas.append(delta)
        change = max(deltas)
    return q

def expected_max_q_value(s,a,q,env):
    Qnext = 0
    qmax = np.max(q,1)
    for prob, snext, rew, terminal in env.P[s][a]:
        Qnext += prob * qmax[snext]
    return Qnext
