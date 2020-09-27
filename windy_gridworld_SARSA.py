import torch
import gym
from windy_gridworld import WindyGridworldEnv
from utils import draw, gen_epsilon_greedy_policy
from collections import defaultdict


def sarsa(env, gamma, n_episode, alpha, gen_policy):
    """Obtain the optimal policy with on-policy SARSA algorithm
    Args:
        env: OpenAI Gym environment
        gamma: discount factor
        n_episode: the number of episode
        alpha: learning rate
        gen_policy: the policy function, which generates action from n_action and Q
    Returns:
        optimal Q, optimal policy, length of each episode, total reward of each episode
    """

    n_action = env.action_space.n
    Q = defaultdict(lambda: torch.zeros(n_action))
    lengths = [0] * n_episode
    rewards = [0] * n_episode

    for episode in range(n_episode):
        state = env.reset()
        is_done = False
        action = gen_policy(n_action, Q)

        while not is_done:
            next_state, reward, is_done, info = env.step(action)
            next_action = gen_policy(next_state, Q)
            td_delta = reward + gamma * Q[next_state][next_action] - Q[state][action]
            Q[state][action] += alpha * td_delta
            lengths[episode] += 1
            rewards[episode] += reward
            if is_done:
                break
            state = next_state
            action = next_action
    
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    
    return Q, policy, lengths, rewards


if __name__ == '__main__':
    env = WindyGridworldEnv()
    n_episode = 500
    gamma = 1
    alpha = 0.4
    epsilon = 0.1
    gen_policy = gen_epsilon_greedy_policy(env.action_space.n, epsilon)
    optimal_Q, optimal_policy, lengths, rewards = sarsa(env, gamma, n_episode, alpha, gen_policy)
    print(optimal_Q, optimal_policy)
    draw('windy_gridworld_lengths_episode.png', 'Length over time', 'Episode', 'Length', lengths)
    draw('windy_gridworld_rewards_episode.png', 'Rewards over time', 'Episode', 'Rewards', rewards)
