import numpy as np
import envs.walled_gridworld as env
import utils
import algorithms.e_greedy_policy as policy
from algorithms.nn_q_function import NNQ
import matplotlib.pyplot as plt


def plot_Q(Q):

    V = [[], [], [], []]
    X = np.arange(0.0, 5.1, 0.1)
    for x in X:
        for y in X:
            vals = Q.compute_all_actions(np.array([x, y]))
            V[0].append(vals[0])
            V[1].append(vals[1])
            V[2].append(vals[2])
            V[3].append(vals[3])
    for v in V:
        v[-1][0] = 0.0
    V = [np.flip(np.array(v).reshape(X.size, X.size), axis=0) for v in V]
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(V[0], cmap="hot", interpolation="gaussian")
    ax[0, 0].set_title("UP")
    ax[0, 1].imshow(V[1], cmap="hot", interpolation="gaussian")
    ax[0, 1].set_title("RIGHT")
    ax[1, 0].imshow(V[2], cmap="hot", interpolation="gaussian")
    ax[1, 0].set_title("DOWN")
    ax[1, 1].imshow(V[3], cmap="hot", interpolation="gaussian")
    ax[1, 1].set_title("LEFT")
    plt.show()


def solve_task(mdp, Q, epsilon=0.2, n_iter=20, n_fit=1, batch_size=1, render=False, verbose=False):
    pi_exp = policy.eGreedyPolicy(Q, Q.actions, epsilon)
    pi_eval = policy.eGreedyPolicy(Q, Q.actions, 0)

    rewards = []
    rewards.append(utils.evaluate_policy(mdp, pi_eval, initial_states=[np.array([0., 0.]) for _ in range(5)], render=render)[0])
    if verbose:
        print("===============================================")
        print("Zero-shot: {0}".format(rewards[-1]))
        print("===============================================")

    dataset = []

    for i in range(n_iter):
        dataset.append(utils.generate_episodes(mdp, pi_exp, batch_size))
        samples = np.concatenate(dataset)
        t, s, a, r, s_prime, absorbing, sa = utils.split_data(samples, 2, 1)

        for k in range(n_fit):
            y = Q.compute_bellman_target(r, s_prime, absorbing)
            Q.fit(s, a, y)

        if verbose:
            plot_Q(Q)

        rewards.append(utils.evaluate_policy(mdp, pi_eval, initial_states=[np.array([0., 0.]) for _ in range(5)],render=render)[0])
        if verbose:
            print("===============================================")
            print("Iteration " + str(i))
            print("Reward: " + str(rewards[-1]))
            print("===============================================")
    return rewards


#size = 5
#n_actions = 4
#mdp = env.WalledGridworld(np.array((size, size)), door_x=2.5)
#Q = NNQ(range(n_actions), 2, gamma=mdp.gamma)
#n_iter = 20
#batch_size = 10
#n_fit = 1
#epsilon = 0.2
#render = False
#verbose = True

#rewards = solve_task(mdp,Q,epsilon,n_iter,n_fit,batch_size,render,verbose)
#print(rewards)
#rewards = np.array(rewards)
#iters = np.arange(rewards.shape[0])
#plt.plot(iters,rewards)
#plt.show()

# Global parameters
size = 5
n_actions = 4
n_source = 5
n_target = 1
n_iter = 50
batch_size = 1
n_fit = 1
epsilon = 0.2
render = False
verbose = True
fit_source = False

if fit_source:
    # Generate the source tasks
    doors = np.random.uniform(0.5,4.5,size=(n_source,))
    print("Source tasks: {0}".format(doors))
    source_mdps = [env.WalledGridworld(np.array((size, size)), door_x=d) for d in doors]
    source_Q = [NNQ(range(n_actions), 2, gamma=mdp.gamma, layers=(16,16,16)) for mdp in source_mdps]

    # Solve the source tasks
    source_rews = [solve_task(mdp,Q,epsilon,n_iter,n_fit,batch_size,render,verbose) for mdp,Q in zip(source_mdps,source_Q)]

    # Compute prior
    weights = np.array([Q.get_weights() for Q in source_Q])
    mean = np.mean(weights, axis=0)
    covariance = np.cov(weights.T, bias=True)
    utils.save_object((mean,covariance), "prior")

# Load prior
prior = utils.load_object("prior")
mean = prior[0]
covariance = prior[1]

# Generate the target task
doors = np.random.uniform(0.5,4.5,size=(n_target,))
print("Target tasks: {0}".format(doors))
target_mdps = [env.WalledGridworld(np.array((size, size)), door_x=d) for d in doors]
target_Q = [NNQ(range(n_actions), 2, gamma=mdp.gamma, layers=(16,16,16), prior_mean=mean, prior_cov=covariance) for mdp in target_mdps]

# Solve target tasks
target_rews = [solve_task(mdp,Q,epsilon,n_iter,n_fit,batch_size,True,verbose) for mdp,Q in zip(target_mdps,target_Q)]

# Plot performance
target_rews = np.array(target_rews)
means = np.mean(target_rews, axis = 0)
stds = np.std(target_rews, axis = 0)
plt.errorbar(np.arange(target_rews.shape[0]),means,stds)
plt.show()



