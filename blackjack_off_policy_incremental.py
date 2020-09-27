import torch
import gym
from collections import defaultdict

env = gym.make('Blackjack-v0')


def gen_random_policy(n_action):
    probs = torch.ones(n_action) / n_action
    def policy_function(state):
        return probs
    return policy_function

random_policy = gen_random_policy(env.action_space.n)


def run_episode(env, behavior_policy):
    r"""Run an episode given a behavior policy.
    Args:
        env: OpenAI Gym environment
        behavior_policy: behavior policy
    Returns:
        resulting states, actions and rewards for the entire episode
    """

    state = env.reset()
    rewards = []
    actions = []
    states = []
    is_done = False

    while not is_done:
        probs = behavior_policy(state)
        action = torch.multinomial(probs, 1).item()
        actions.append(action)
        states.append(state)
        state, reward, is_done, info = env.step(action)
        rewards.append(reward)
        if is_done:
            break
    
    return states, actions, rewards


def mc_control_off_policy_incremental(env, gamma, n_episode, behavior_policy):
    """Obtain the optimal policy with off-policy Monte Carlo control with incremental way of updating the Q function
    Args:
        env: OpenAI Gym environment
        gamma: discount factor
        n_episode: number of episodes
        behavior_policy: behavior policy
    Returns:
        the optimal Q-function, and optimal policy
    """

    n_action = env.action_space.n
    number = defaultdict(int)
    Q = defaultdict(lambda: torch.empty(n_action))

    for episode in range(n_episode):
        weight = 1
        states_t, actions_t, rewards_t = run_episode(env, behavior_policy)
        return_t = 0
        
        for state_t, action_t, reward_t in zip(states_t[::-1], actions_t[::-1], rewards_t[::-1]):
            return_t = gamma * return_t + reward_t
            number[(state_t, action_t)] += 1
            Q[state_t][action_t] += (weight / number[(state_t, action_t)]) * (return_t - Q[state_t][action_t])
            if action_t != torch.argmax(Q[state_t]).item():
                break
            weight *= 1. / behavior_policy(state_t)[action_t]
        
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()

    return Q, policy


gamma = 1
n_episode = 500000

optimal_Q, optimal_policy = mc_control_off_policy_incremental(env, gamma, n_episode, random_policy)

optimal_value = defaultdict(float)
for state, action_values in optimal_Q.items():
    optimal_value[state] = torch.max(action_values).item()

print('Optimal Q:\n', optimal_Q)
print('Optimal policy:\n', optimal_policy)
print('Optimal value:\n', optimal_value)