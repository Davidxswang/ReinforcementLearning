import gym
import torch
import time
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

n_state = env.observation_space.shape[0]
n_action = env.action_space.n

best_weight = torch.rand(n_state, n_action)
best_total_reward = 0

total_rewards = []
total_rewards_eval = []

n_episode = 1000
n_episode_eval = 100

noise_scale = 0.01
noise_scale_lower_bound = 0.0001
noise_scale_upper_bound = 2

def run_episode(env, weight):
    state = env.reset()
    total_reward = 0
    is_done = False

    while not is_done:
        state = torch.from_numpy(state).float()
        action = torch.argmax(torch.matmul(state, weight)).item()
        state, reward, is_done, _ = env.step(action)
        total_reward += reward
    return total_reward


if __name__ == '__main__':
    for i in range(n_episode):
        weight = best_weight + noise_scale * torch.rand(n_state, n_action)
        total_reward = run_episode(env, weight)
        print(f"\repisode {i}, total reward: {total_reward:.1f}", end='')
        total_rewards.append(total_reward)
        if total_reward >= best_total_reward:
            best_total_reward = total_reward
            best_weight = weight
            noise_scale = max(noise_scale_lower_bound, noise_scale / 2)
        else:
            noise_scale = min(noise_scale_upper_bound, noise_scale * 2)
        if i >= 99 and sum(total_rewards[-100:]) >= 19500:
            break
    
    print(f"\nAverage total reward over {n_episode} episodes: {sum(total_rewards) / len(total_rewards):.2f}")

    for i in range(n_episode_eval):
        total_reward = run_episode(env, best_weight)
        print(f"\rtotal_reward: {total_reward} in eval episode {i}", end="")
        total_rewards_eval.append(total_reward)

    print(f"\nAverage total reward using the best weight over {n_episode_eval} eval episodes: {sum(total_rewards_eval) / n_episode_eval}")

    plt.plot(total_rewards)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.show()
    plt.savefig('hill_climbing.png')