import gym
import torch
from torch import nn
from policy_network import PolicyNetwork
from utils import draw


def reinforce(env, estimator, n_episode, gamma=1.0):
    """Reinforce algorithm

    Args:
        env: OpenAI gym environment
        estimator: policy network
        n_episode: number of episodes
        gamma: the discount factor
    Returns:
        total reward for each episode (a list)
    """
    total_reward_episode = [0] * n_episode

    for episode in range(n_episode):
        log_probs = []
        rewards = []
        state = env.reset()

        while True:
            action, log_prob = estimator.get_action(state)
            next_state, reward, is_done, _ = env.step(action)
            total_reward_episode[episode] += reward
            log_probs.append(log_prob)
            rewards.append(reward)

            if is_done:
                returns = []
                Gt = 0
                power = 0
                for reward in reversed(rewards):
                    Gt += gamma ** power * reward
                    power += 1
                    returns.append(Gt)
                returns = returns[::-1]
                returns = torch.tensor(returns)
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)
                estimator.update(returns, log_probs)
                print(f"Episode {episode}, total reward {total_reward_episode[episode]}")
                break
            
            state = next_state
    
    return total_reward_episode


if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.n
    n_hidden = 128
    lr = 0.003

    policy_net = PolicyNetwork(n_state, n_action, n_hidden, lr)

    gamma = 0.9
    n_episode = 500
    
    rewards = reinforce(env, policy_net, n_episode, gamma)

    draw("cartpole_reinforce.png", "Rewards over time", "episode", "reward", rewards)