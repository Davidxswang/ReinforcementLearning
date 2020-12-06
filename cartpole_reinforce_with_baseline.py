import gym
import torch
from torch import nn
from policy_network import PolicyNetwork, ValueNetwork
from utils import draw


def reinforce(env, estimator_policy, estimator_value, n_episode, gamma=1.0):
    """Reinforce algorithm with baseline

    Args:
        env: OpenAI gym environment
        estimator_policy: policy network
        estimator_value: value network
        n_episode: number of episodes
        gamma: the discount factor
    Returns:
        total reward for each episode (a list)
    """
    total_reward_episode = [0] * n_episode

    for episode in range(n_episode):
        log_probs = []
        states = []
        rewards = []
        state = env.reset()

        while True:
            states.append(state)
            action, log_prob = estimator_policy.get_action(state)
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
                baseline = estimator_value.predict(states)
                advantages = returns - baseline
                estimator_policy.update(advantages, log_probs)
                estimator_value.update(states, returns)
                print(f"Episode {episode}, total reward {total_reward_episode[episode]}")
                break
            
            state = next_state
    
    return total_reward_episode


if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.n
    
    n_hidden_p = 64
    lr_p = 0.003
    policy_net = PolicyNetwork(n_state, n_action, n_hidden_p, lr_p)

    n_hidden_v = 64
    lr_v = 0.003
    value_net = ValueNetwork(n_state, n_hidden_v, lr_v)

    gamma = 0.9
    n_episode = 2000
    
    rewards = reinforce(env, policy_net, value_net, n_episode, gamma)

    draw("cartpole_reinforce_with_baseline.png", "Rewards over time", "episode", "reward", rewards)