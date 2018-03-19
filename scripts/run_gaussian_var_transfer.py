import numpy as np
import envs.walled_gridworld as wgw
import VariationalTransfer.LinearQRegressor as linq
import VariationalTransfer.BellmanOperator as bellmanop
import VariationalTransfer.MellowBellmanOperator as mmbellman
import VariationalTransfer.VarTransfer as vartrans
import VariationalTransfer.Distributions as dist
import features.agrbf as rbf
import utils
import algorithms.e_greedy_policy as policy
import algorithms.regularized_lsvi as lsvi
from scripts.run_nfqi import plot_Q


"""
    FQI with linear regressor for Q(s,a) 
"""
def linearFQI(mdp, Q, epsilon=0, n_iter=1, batch_size=1, render=False, verbose=False, n_fit=20, bellman_operator=None):
    pol = policy.eGreedyPolicy(Q, Q.actions, epsilon)
    pol_g = policy.eGreedyPolicy(Q, Q.actions, 0)
    r = list()
    n_act = len(Q.actions)

    rew, _, _, _ = utils.evaluate_policy(mdp, pol_g, n_episodes=5, initial_states=np.array([0., 0.]), render=render)
    r.append(rew)
    utils.plot_Q(Q, size=tuple(mdp.size))

    samples = utils.generate_episodes(mdp, pol, batch_size, render=False)
    feat = Q.compute_features(samples[:, 1:])

    if bellman_operator is None:
        bellman = bellmanop.BellmanOperator(Q)
    else:
        bellman_operator.set_Q(Q)
        bellman = bellman_operator

    for i in range(n_iter):
        new_samples = utils.generate_episodes(mdp, pol, batch_size, render=False)
        samples = np.vstack((samples, new_samples))
        feat = np.vstack((feat, Q.compute_features(new_samples[:, 1:])))

        for k in range(n_fit):
            targets = bellman(samples[:, 1:])
            w = lsvi.RegularizedLSVI.solve(feat, targets, prior=False)
            Q.update_weights(w)

        if render:
            mdp._render(close=True)

        utils.plot_Q(Q, size=tuple(mdp.size))
        rew, _, _, _ = utils.evaluate_policy(mdp, pol_g, n_episodes=5, initial_states=np.array([0., 0.]), render=render)
        r.append(rew)
        if verbose:
            print("===============================================")
            print("Iteration " + str(i))
            print("Reward: " + str(rew))
            print("===============================================")

    return r



if __name__ == "__main__":

    n = 5
    N = 36
    acts = 4
    state_dim = 2
    action_dim = 1
    k = N*acts

    x = np.linspace(0, n, np.sqrt(N))
    y = np.linspace(0, n, np.sqrt(N))
    a = np.linspace(0, acts-1, acts)
    mean_x, mean_y, mean_a = np.meshgrid(x,y,a)
    mean = np.hstack((mean_x.reshape(k,1), mean_y.reshape(k,1), mean_a.reshape(k,1)))

    state_var = (n/((x.shape[0]-1)*3))**2
    action_var = 0.1**2
    covar = np.eye(state_dim + action_dim)
    covar[0:state_dim, 0:state_dim] *= state_var
    covar[-1, -1] *= action_var
    covar = np.tile(covar, (k, 1))

    # features
    features = rbf.AGaussianRBF(mean, covar, K=k, dims=state_dim+action_dim)
    sources = list()
    q_functions = list()

    # Source tasks
    for i in range(n+1):
        sources.append(wgw.WalledGridworld(np.array((n, n)), door_x=i+.5))
        q_functions.append(linq.LinearQRegressor(features, np.arange(acts), state_dim, action_dim))
        linearFQI(sources[i], q_functions[i], epsilon=0.2, n_iter=20, render=False, verbose=True)


    # # Create Target task
    # world = wgw.WalledGridworld(np.array([n, n]), door_x=np.random.ranf(1)[0]*(n-0.5) + 0.5)
    # q = linq.LinearQRegressor(features, np.arange(acts), state_dim, action_dim)
    # prior = dist.AnisotropicNormalPosterior(np.zeros(k), np.ones(k))
    # bellman = mmbellman.LinearQMellowBellman(q, gamma=world.gamma)
    # var = vartrans.VarTransferGaussian(world, bellman, prior, 1e-3, 1e-3)
    # var.solve_task(verbose=True)