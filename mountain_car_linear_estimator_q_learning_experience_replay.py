import torch
import gym
import random
from collections import deque
from linear_estimator import LinearEstimator
from utils import gen_epsilon_greedy_policy, draw


def q_learning_with_replay(env, estimator, n_episode, n_action, replay_size, gamma=1.0, epsilon=0.1, epsilon_decay=0.99):
    r"""
    Q-Learning using Function Approximation with experience replay

    Args:
        env: OpenAI environment
        estimator: linear estimator
        n_episode: number of episode
        n_action: number of action
        replay_size: the size of replay sampling space
        gamma: discount factor
        epsilon: controls the trade-off between exploration and exploitation
        epsilon_decay: decrease the epsilon over time
    Returns:
        list: total reward over episodes
    """
    total_reward_episode = [0.0] * n_episode
    memory = deque(maxlen=400)

    for episode in range(n_episode):
        policy = gen_epsilon_greedy_policy(n_action, epsilon * epsilon_decay ** episode, estimator)
        state = env.reset()
        is_done = False    

        while not is_done:
            action = policy(state)
            next_state, reward, is_done, info = env.step(action)
            total_reward_episode[episode] += reward
            if is_done:
                break

            q_values_next = estimator.predict(next_state)
            target = reward + gamma * torch.max(q_values_next)
            memory.append((state, action, target))
            state = next_state

        replay_data = random.sample(memory, min(len(memory), replay_size))

        for state, action, target in replay_data:
            estimator.update(state, action, target)
    
    return total_reward_episode


if __name__ == '__main__':
    env = gym.envs.make('MountainCar-v0')

    n_state = env.observation_space.shape[0]
    n_action = env.action_space.n
    n_feature = 200
    lr = 0.03
    estimator = LinearEstimator(n_feature, n_state, n_action, lr)

    n_episode = 1500
    replay_size = 190

    total_reward_episode = q_learning_with_replay(env, estimator, n_episode, n_action, replay_size, gamma=1.0, epsilon=0.1, epsilon_decay=0.99)

    draw('mountain_car_linear_estimator_q_learning_experience_replay.png', 'Reward over episodes', 'episode', 'reward', total_reward_episode)