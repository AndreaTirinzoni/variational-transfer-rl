import numpy as np
import matplotlib.pyplot as plt
import envs.walled_gridworld as wgw
import VariationalTransfer as vt
import features.agrbf as rbf
import utils

if __name__ == "__main__":

    n = 5
    N = 36
    acts = 4
    state_dim = 2
    action_dim = 1
    k = N*acts

    x = np.linspace(0, n, np.sqrt(N))
    y = np.linspace(0, n, np.sqrt(N))
    a = np.linspace(0, acts, acts)
    mean_x, mean_y, mean_a = np.meshgrid(x,y,a)
    mean = np.hstack((mean_x.reshape(k,1), mean_y.reshape(k,1), mean_a.reshape(k,1)))

    state_var = (n/(x.shape[0]*3))**2
    action_var = 0.1**2
    covar = np.eye(state_dim + action_dim)
    covar[0:state_dim, 0:state_dim] *= state_var
    covar[-1, -1] *= action_var
    covar = np.tile(covar, (k, 1))

    # features
    features = rbf.AGaussianRBF(mean, covar, K=k, dims=state_dim+action_dim)
    sources = list()

    # Source tasks
    for i in range(n+1):
        sources.append(wgw.WalledGridworld(np.array((n, n)), door_x=i-0.5))

    # Create Target task
    world = wgw.WalledGridworld(np.array([n, n]), door_x=np.random.ranf(1)[0]*(n-0.5) + 0.5)
