import torch
import gym
from utils import draw, gen_epsilon_greedy_policy


def double_q_learning(env, gamma, n_episode, alpha, gen_policy):
    """Obtain the optimal policy with off-policy double Q-learning
    Args:
        env: OpenAI Gym environment
        gamma: discount factor
        n_episode: the number of episode
        alpha: learning rate
        gen_policy: a policy function will take state and Q, and generate action
    Returns:
        optimal Q, policy, lengths over episodes, rewards over episodes
    """
    n_action = env.action_space.n
    n_state = env.observation_space.n
    Q1 = torch.zeros(n_state, n_action)
    Q2 = torch.zeros(n_state, n_action)
    lengths = [0] * n_episode
    rewards = [0] * n_episode

    for episode in range(n_episode):
        state = env.reset()
        is_done = False
        
        while not is_done:
            action = gen_policy(state, Q1 + Q2)
            next_state, reward, is_done, info = env.step(action)
            if torch.rand(1).item() < 0.5:
                next_action = torch.argmax(Q1[next_state])
                td_delta = reward + gamma * Q2[next_state][next_action] - Q1[state][action]
                Q1[state][action] += alpha * td_delta
            else:
                next_action = torch.argmax(Q2[next_state])
                td_delta = reward + gamma * Q1[next_state][next_action] - Q2[state][action]
                Q2[state][action] += alpha * td_delta
            lengths[episode] += 1
            rewards[episode] += reward
            if is_done:
                break
            state = next_state
        
    policy = {state:torch.argmax(Q1[state]+Q2[state]).item() for state in range(n_state)}
    return Q1+Q2, policy, lengths, rewards


if __name__ == '__main__':
    env = gym.make('Taxi-v3')
    n_episode = 3000
    gamma = 1
    alpha = 0.4
    epsilon = 0.1
    gen_policy = gen_epsilon_greedy_policy(env.action_space.n, epsilon)
    Q, policy, lengths, rewards = double_q_learning(env, gamma, n_episode, alpha, gen_policy)
    print(Q)
    print(policy)
    draw('taxi_double_q_learning_lengths.png', 'Lengths over time', 'Episode', 'Length', lengths)
    draw('taxi_double_q_learning_rewards.png', 'Rewards over time', 'Episode', 'reward', rewards)