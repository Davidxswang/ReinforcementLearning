import gym
import torch
from policy_network import Estimator
from utils import draw


def cross_entropy(env, estimator, n_episode, n_samples):
    """Cross-entropy algorithm for policy learning

    Args:
        env: OpenAI gym environment
        estimator: binary estimator
        n_episode: number of episodes
        n_samples: number of training samples to use
    """
    experience = []
    
    for episode in range(n_episode):
        rewards = 0
        actions = []
        states = []
        state = env.reset()

        while True:
            action = env.action_space.sample()
            states.append(state)
            actions.append(action)
            next_state, reward, is_done, _ = env.step(action)
            rewards += reward

            if is_done:
                for state, action in zip(states, actions):
                    experience.append((rewards, state, action))
                break
                
            state = next_state
    
    experience = sorted(experience, key=lambda x:x[0], reverse=True)
    selected_experience = experience[:n_samples]
    train_states = [exp[1] for exp in selected_experience]
    train_actions = [exp[2] for exp in selected_experience]

    for _ in range(100):
        estimator.update(train_states, train_actions)


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    n_state = env.observation_space.shape[0]
    lr = 0.01

    estimator = Estimator(n_state, lr)

    n_episode = 5000
    n_samples = 10000
    
    cross_entropy(env, estimator, n_episode, n_samples)

    n_test_episode = 100
    total_reward_episode = [0] * n_test_episode
    
    for episode in range(n_test_episode):
        state = env.reset()
        is_done = False

        while not is_done:
            action = 1 if estimator.predict(state).cpu().item() >= 0.5 else 0
            next_state, reward, is_done, _ = env.step(action)
            total_reward_episode[episode] += reward
            state = next_state
        print(f"Episode: {episode}, total reward: {total_reward_episode[episode]}")
        
    draw("cartpole_cross_entropy.png", "Rewards over time", "episode", "reward", total_reward_episode)