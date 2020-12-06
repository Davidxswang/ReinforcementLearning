import gym
import torch
from policy_network import ActorCriticGaussianPolicyNetwork
from utils import draw
import sklearn.preprocessing
import numpy as np


def actor_critic(env, estimator, n_episode, gamma=1.0, scale_state=None):
    """Continuous actor-critic algorithm

    Args:
        env: OpenAI gym environment
        estimator: policy network
        n_episode: number of episodes
        gamma: the discount factor
        scale_state: the function to scale the state
    Returns:
        total reward for each episode (a list)
    """
    total_reward_episode = [0] * n_episode

    for episode in range(n_episode):
        log_probs = []
        rewards = []
        state_values = []
        state = env.reset()

        while True:
            state = scale_state(state)
            action, log_prob, state_value = estimator.get_action(state)
            action = action.clip(env.action_space.low[0], env.action_space.high[0])
            next_state, reward, is_done, _ = env.step(action)
            total_reward_episode[episode] += reward
            log_probs.append(log_prob)
            state_values.append(state_value)
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
                estimator.update(returns, log_probs, state_values)
                print(f"Episode {episode}, total reward {total_reward_episode[episode]}")
                break
            
            state = next_state
    
    return total_reward_episode    



if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')

    state_space_samples = np.array([env.observation_space.sample() for x in range(10000)])
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(state_space_samples)

    def scale_state(state):
        scaled = scaler.transform([state])
        return scaled[0]

    n_state = env.observation_space.shape[0]
    n_action = 1
    n_hidden = 128
    lr = 0.003

    policy_net = ActorCriticGaussianPolicyNetwork(n_state, n_action, n_hidden, lr)

    gamma = 0.9
    n_episode = 200
    
    rewards = actor_critic(env, policy_net, n_episode, gamma, scale_state)

    draw("continuous_mountain_car_advantage_actor_critic.png", "Rewards over time", "episode", "reward", rewards)