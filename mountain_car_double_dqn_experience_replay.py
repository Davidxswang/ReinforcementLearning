import torch
import gym
import random
from collections import deque
from dqn import Double_DQN
from utils import draw, gen_epsilon_greedy_policy


def q_learning(env, estimator, n_episode, n_action, replay_size, target_update=10, gamma=1.0, epsilon=0.1, epsilon_decay=0.99):
    """Deep Q-Learning using double DQN, with experience replay

    Args:
        env: OpenAI environment
        estimator: the estimator to learn to predict the Q values from state
        n_episode: number of episode
        n_action: number of action
        replay_size: the number of samples used for replay
        target_update: the frequency of updating target model
        gamma: discount factor
        epsilon: to control the trade-off between exploration and exploitation
        epsilon_decay: to control the epsilon over time
    """
    memory = deque(maxlen=10000)
    total_reward_episode = [0] * n_episode

    for episode in range(n_episode):
        if episode % target_update == 0:
            estimator.copy_target()
        policy = gen_epsilon_greedy_policy(n_action, epsilon, estimator)
        state = env.reset()
        is_done = False
        
        while not is_done:
            action = policy(state)
            next_state, reward, is_done, _ = env.step(action)
            total_reward_episode[episode] += reward
            modified_reward = next_state[0] + 0.5
            if next_state[0] >= 0.5:
                modified_reward += 100
            elif next_state[0] >= 0.25:
                modified_reward += 20
            elif next_state[0] >= 0.1:
                modified_reward += 10
            elif next_state[0] >= 0:
                modified_reward += 5
            
            memory.append((state, action, next_state, modified_reward, is_done))

            if is_done:
                break

            estimator.replay(memory, replay_size, gamma)
            state = next_state

        print(f"Episode {episode}, total reward {total_reward_episode[episode]}, epsilon {epsilon}")
        epsilon = max(epsilon * epsilon_decay, 0.01)
    
    return total_reward_episode


if __name__ == '__main__':
    env = gym.envs.make("MountainCar-v0")

    n_state = env.observation_space.shape[0]
    n_action = env.action_space.n
    n_hidden = 50
    lr = 0.01

    dqn = Double_DQN(n_state, n_action, n_hidden, lr)
    
    n_episode = 1000
    replay_size = 20
    target_update = 10
    rewards = q_learning(env, dqn, n_episode, n_action, replay_size, target_update, gamma=0.9, epsilon=0.3, epsilon_decay=0.99)
    
    draw("mountain_car_double_dqn_experience_replay.png", "Total Rewards over Episodes", "Episode", "Reward", rewards)