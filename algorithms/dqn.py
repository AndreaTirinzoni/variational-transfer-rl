from approximators.mlp_torch import MLPQFunction
from operators.dqn import DQNOperator

# Initialization of DQN elements
def DQN(state_dim, action_dim, n_actions, gamma, layers=(32,), initial_params=None, target_update_freq=500):

    Q = MLPQFunction(state_dim, n_actions, layers=layers, initial_params=initial_params)
    Q_target = MLPQFunction(state_dim, n_actions, layers=tuple(layers), initial_params=initial_params)

    if initial_params is None:
        Q_target._w = Q._w
    operator = DQNOperator(state_dim, action_dim, gamma, Q_target, target_update_freq)

    return Q, operator