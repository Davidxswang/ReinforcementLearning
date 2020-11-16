import torch
import gym
from dqn import DQN
from utils import draw, gen_epsilon_greedy_policy


def q_learning(env, estimator, n_episode, n_action, gamma=1.0, epsilon=0.1, epsilon_decay=0.99):
    """Deep Q-Learning using DQN

    Args:
        env: OpenAI environment
        estimator: the estimator to learn to predict the Q values from state
        n_episode: number of episode
        n_action: number of action
        gamma: discount factor
        epsilon: to control the trade-off between exploration and exploitation
        epsilon_decay: to control the epsilon over time
    Returns:
        list: total rewards for each of the episode
    """
    total_reward_episode = [0] * n_episode

    for episode in range(n_episode):
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
            
            # get the q_values, pay attention to [0]
            q_values = estimator.predict(state).tolist()[0]
            
            if is_done:
                q_values[action] = modified_reward
                estimator.update(state, q_values)
                break

            q_values_next = estimator.predict(next_state)
            q_values[action] = modified_reward + gamma * torch.max(q_values_next).item()
            estimator.update(state, q_values)
            state = next_state
        print(f"Episode {episode}, total reward {total_reward_episode[episode]}, epsilon {epsilon}")
        epsilon = max(epsilon * epsilon_decay, 0.01)
    
    return total_reward_episode


if __name__ == '__main__':
    env = gym.envs.make("MountainCar-v0")

    n_state = env.observation_space.shape[0]
    n_action = env.action_space.n
    n_hidden = 50
    lr = 0.001
    n_episode = 1000

    dqn = DQN(n_state, n_action, n_hidden, lr)
    rewards = q_learning(env, dqn, n_episode, n_action, gamma=0.99, epsilon=0.3, epsilon_decay=0.99)

    draw("mountain_car_dqn.png", "Total Rewards over Episodes", "Episode", "Reward", rewards)