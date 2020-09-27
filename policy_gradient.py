import gym
import torch
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

n_state = env.observation_space.shape[0]
n_action = env.action_space.n


def run_episode(env, weight):
    state = env.reset()
    grads = []
    total_reward = 0
    is_done = False
    
    while not is_done:
        state = torch.from_numpy(state).float()
        z = torch.matmul(state, weight)
        probs = torch.nn.Softmax()(z)
        action = int(torch.bernoulli(probs[1]).item())
        d_softmax = torch.diag(probs) - probs.view(-1, 1) * probs
        d_log = d_softmax[action] / probs[action]
        grad = state.view(-1, 1) * d_log
        grads.append(grad)
        state, reward, is_done, _ = env.step(action)
        total_reward += reward
    
    return total_reward, grads


n_episode = 1000

weight = torch.rand(n_state, n_action)

total_rewards = []

learning_rate = 0.001

for episode in range(n_episode):
    total_reward, gradients = run_episode(env, weight)
    print(f"episode {episode} total reward: {total_reward}")
    for i, gradient in enumerate(gradients):
        weight += learning_rate * gradient * (total_reward - i)
    total_rewards.append(total_reward)
    if episode >= 99 and sum(total_rewards[-100:]) >= 19500:
        break

print(f"Average total reward over {n_episode} episode: {sum(total_rewards) / episode+1}")

plt.plot(total_rewards)
plt.xlabel('episode')
plt.ylabel('reward')
plt.savefig('policy_gradient.png')

n_episode_eval = 100
total_rewards_eval = []

for episode in range(n_episode_eval):
    total_reward, _ = run_episode(env, weight)
    print(f"eval episode {episode} total reward: {total_reward}")
    total_rewards_eval.append(total_reward)

print(f"Average total reward over {n_episode_eval} eval episode: {sum(total_rewards_eval) / n_episode_eval}")