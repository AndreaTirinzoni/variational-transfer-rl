import numpy as np
from policies import EpsilonGreedy, ScheduledEpsilonGreedy
from approximators.mlp_torch import MLPQFunction
from buffer import Buffer
import utils
import time

from operators.dqn import DQNOperator


def learn(mdp,
          Q,
          operator,
          max_iter=5000,
          buffer_size=10000,
          batch_size=50,
          alpha=0.001,
          train_freq=1,
          eval_freq=50,
          eps_start=1.0,
          eps_end=0.02,
          exploration_fraction=0.2,
          random_episodes=0,
          eval_states=None,
          eval_episodes=1,
          mean_episodes=50,
          preprocess=lambda x: x,
          seed=None,
          render=False,
          verbose=True):
    if seed is not None:
        np.random.seed(seed)

    # Randomly initialize the weights in case an MLP is used
    if isinstance(Q, MLPQFunction):
        # Q.init_weights()
        if isinstance(operator, DQNOperator):
            operator._q_target._w = Q._w

    # Initialize policies
    schedule = np.linspace(eps_start, eps_end, exploration_fraction * max_iter)
    pi = ScheduledEpsilonGreedy(Q, np.arange(mdp.action_space.n), schedule)
    pi_u = EpsilonGreedy(Q, np.arange(mdp.action_space.n), epsilon=1)
    pi_g = EpsilonGreedy(Q, np.arange(mdp.action_space.n), epsilon=0)

    # Add random episodes if needed
    init_samples = utils.generate_episodes(mdp, pi_u, n_episodes=random_episodes,
                                           preprocess=preprocess) if random_episodes > 0 else None
    if random_episodes > 0:
        t, s, a, r, s_prime, absorbing, sa = utils.split_data(init_samples, mdp.state_dim, mdp.action_dim)
        init_samples = np.concatenate((t[:, np.newaxis], preprocess(s), a, r[:, np.newaxis], preprocess(s_prime),
                                       absorbing[:, np.newaxis]), axis=1)

    # Figure out the effective state-dimension after preprocessing is applied
    eff_state_dim = preprocess(np.zeros(mdp.state_dim)).size

    # Create replay buffer
    buffer = Buffer(buffer_size, eff_state_dim)
    n_init_samples = buffer.add_all(init_samples) if random_episodes > 0 else 0

    # Results
    iterations = []
    episodes = []
    n_samples = []
    evaluation_rewards = []
    learning_rewards = []
    episode_rewards = [0.0]
    episode_t = []
    l_2 = []
    l_inf = []

    # Adam initial params
    m_t = 0
    v_t = 0
    t = 0

    # Init env
    s = mdp.reset()
    h = 0

    start_time = time.time()

    # Learning
    for i in range(max_iter):

        # Take epsilon-greedy action wrt current Q-function
        s_prep = preprocess(s)
        a = pi.sample_action(s_prep)
        # Step
        s_prime, r, done, _ = mdp.step(a)
        # Build the new sample and add it to the dataset
        buffer.add_sample(h, s_prep, a, r, preprocess(s_prime), done)

        # Take a step of gradient if needed
        if i % train_freq == 0:
            # Estimate gradient
            g = operator.gradient_be(Q, buffer.sample_batch(batch_size))
            # Take a gradient step
            Q._w, t, m_t, v_t = utils.adam(Q._w, g, t, m_t, v_t, alpha=alpha)

        # Add reward to last episode
        episode_rewards[-1] += r * mdp.gamma ** h

        s = s_prime
        h += 1
        if done or h >= mdp.horizon:
            episode_rewards.append(0.0)
            s = mdp.reset()
            h = 0
            episode_t.append(i)

        # Evaluate model
        if i % eval_freq == 0:

            # Evaluate greedy policy
            rew = utils.evaluate_policy(mdp, pi_g, render=render, initial_states=eval_states,
                                        n_episodes=eval_episodes, preprocess=preprocess)[0]
            learning_rew = np.mean(episode_rewards[-mean_episodes - 1:-1]) if len(episode_rewards) > 1 else 0.0
            br = operator.bellman_residual(Q, buffer.sample_batch(batch_size)) ** 2
            l_2_err = np.average(br)
            l_inf_err = np.max(br)

            # Append results
            iterations.append(i)
            episodes.append(len(episode_rewards) - 1)
            n_samples.append(n_init_samples + i + 1)
            evaluation_rewards.append(rew)
            learning_rewards.append(learning_rew)
            l_2.append(l_2_err)
            l_inf.append(l_inf_err)

            # Make sure we restart from s
            mdp.reset(s)

            end_time = time.time()
            elapsed_time = end_time - start_time
            start_time = end_time

            if verbose:
                print("Iter {} Episodes {} Rew(G) {} Rew(L) {} L2 {} L_inf {} time {:.1f} s".format(
                    i, episodes[-1], rew, learning_rew, l_2_err, l_inf_err, elapsed_time))
        # if np.mean(episode_rewards[-mean_episodes - 1:-1]) > -80:
        #     render=True


    run_info = [iterations, episodes, n_samples, learning_rewards, evaluation_rewards, l_2, l_inf, episode_rewards[:len(episode_t)], episode_t]
    weights = np.array(Q._w)

    return [mdp.get_info(), weights, run_info]
