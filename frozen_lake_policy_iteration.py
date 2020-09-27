import torch
import gym


def policy_evaluation(env, policy, gamma, threshold):
    """
    Perform policy evaluation
    Args:
        env: OpenAI Gym environment
        policy (Tensor): policy matrix, shape: n_state
        gamma (float): discount factor
        threshold (float): control when to stop iteration
    Returns:
        values of the given policy
    """

    n_state = env.observation_space.n
    value = torch.zeros(n_state)
    
    while True:
        value_temp = torch.zeros(n_state)
        for state in range(n_state):
            action = policy[state].item()
            for trans_prob, new_state, reward, _ in env.env.P[state][action]:
                value_temp[state] += trans_prob * (reward + gamma * value[new_state])
        max_delta = torch.max(torch.abs(value - value_temp))
        value = value_temp.clone()
        if max_delta <= threshold:
            break
    
    return value


def policy_improvement(env, value, gamma):
    """
    Obtain an improved policy based on the values
    Args:
        env: OpenAI Gym environment
        value: the value based on the previous policy
        gamma: the discount factor
    Returns:
        the improved policy
    """

    n_state = env.observation_space.n
    n_action = env.action_space.n
    policy = torch.zeros(n_state)
    for state in range(n_state):
        v_actions = torch.zeros(n_action)
        for action in range(n_action):
            for trans_prob, new_state, reward, _ in env.env.P[state][action]:
                v_actions[action] += trans_prob * (reward + gamma * value[new_state])
        policy[state] = torch.argmax(v_actions)
    
    return policy


def policy_iteration(env, gamma, threshold):
    """
    Solve a given environment with policy iteration algorithm
    Args:
        env: OpenAI Gym environment
        gammas (float): the discount factor
        threshold (float): control when to stop the iteration
    Returns:
        optimal value and optimal policy
    """

    n_state = env.observation_space.n
    n_action = env.action_space.n
    policy = torch.randint(high=n_action, size=(n_state, )).float()


    while True:
        value = policy_evaluation(env, policy, gamma, threshold)
        policy_improved = policy_improvement(env, value, gamma)
        if torch.equal(policy, policy_improved):
            return value, policy
        policy = policy_improved
        


env = gym.make('FrozenLake-v0')

gamma = 0.99
threshold = 0.0001

value, policy = policy_iteration(env, gamma, threshold)
print(value)
print(policy)