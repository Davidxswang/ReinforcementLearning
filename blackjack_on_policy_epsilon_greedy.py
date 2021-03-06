import torch
import gym
from collections import defaultdict

env = gym.make('Blackjack-v0')


def run_episode(env, Q, epsilon, n_action):
    r"""Run an episode given a Q-function.
    Args:
        env: OpenAI Gym environment
        Q: Q-function
        epsilon: trade-off between exploration and exploitation
        n_action: action space
    Returns:
        resulting states, actions and rewards for the entire episode
    """

    state = env.reset()
    rewards = []
    actions = []
    states = []
    is_done = False

    while not is_done:
        probs = torch.ones(n_action) * epsilon / n_action
        best_action = torch.argmax(Q[state]).item()
        probs[best_action] += 1 - epsilon
        action = torch.multinomial(probs, 1).item()
        actions.append(action)
        states.append(state)
        state, reward, is_done, info = env.step(action)
        rewards.append(reward)
        if is_done:
            break
    
    return states, actions, rewards


def mc_control_epsilon_greedy(env, gamma, n_episode, epsilon):
    """Obtain the optimal policy with on-policy Monte Carlo control with epsilon greedy
    Args:
        env: OpenAI Gym environment
        gamma: discount factor
        n_episode: number of episodes
        epsilon: trade-off between exploration and exploitation
    Returns:
        the optimal Q-function, and optimal policy
    """

    n_action = env.action_space.n
    G_sum = defaultdict(float)
    number = defaultdict(int)
    Q = defaultdict(lambda: torch.empty(n_action))

    for episode in range(n_episode):
        states_t, actions_t, rewards_t = run_episode(env, Q, epsilon, n_action)
        return_t = 0
        G = {}
        
        for state_t, action_t, reward_t in zip(states_t[::-1], actions_t[::-1], rewards_t[::-1]):
            return_t = gamma * return_t + reward_t
            G[(state_t, action_t)] = return_t
        
        for state_action, return_t in G.items():
            state, action = state_action
            if state[0] <= 21:
                G_sum[state_action] += return_t
                number[state_action] += 1
                Q[state][action] = G_sum[state_action] / number[state_action]
        
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()

    return Q, policy


gamma = 1
n_episode = 500000
epsilon = 0.1

optimal_Q, optimal_policy = mc_control_epsilon_greedy(env, gamma, n_episode, epsilon)

optimal_value = defaultdict(float)
for state, action_values in optimal_Q.items():
    optimal_value[state] = torch.max(action_values).item()

print('Optimal Q:\n', optimal_Q)
print('Optimal policy:\n', optimal_policy)
print('Optimal value:\n', optimal_value)