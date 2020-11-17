import torch
import gym
import random
from torchvision import transforms as T
from PIL import Image
from collections import deque
import copy
from dqn import Double_DQN
from utils import draw, gen_epsilon_greedy_policy


def get_state(obs, image_size):
    transform = T.Compose([
        T.ToPILImage(),
        T.Grayscale(num_output_channels=1),
        T.Resize((image_size, image_size), interpolation=Image.CUBIC),
        T.ToTensor()
    ])
    state = obs.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    state = transform(state)
    return state


def get_action(action):
    ACTIONS = [0, 2, 3]
    return ACTIONS[action]


def q_learning(env, estimator, n_episode, n_action, replay_size, target_update, image_size, gamma=1.0, epsilon=0.1,  epsilon_decay=0.99):
    """Deep Q-Learning using double DQN, with experience replay

    Args:
        env: OpenAI environment
        estimator: the estimator to learn to predict the Q values from state
        n_episode: number of episode
        n_action: number of action
        replay_size: the number of samples used for replay
        target_update: the frequency of updating target model
        image_size: the height or width of the image
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
        observation = env.reset()
        state = get_state(observation, image_size).view(image_size * image_size).tolist()
        is_done = False
        
        while not is_done:
            action = policy(state)
            next_observation, reward, is_done, _ = env.step(get_action(action))
            total_reward_episode[episode] += reward
            next_state = get_state(next_observation, image_size).view(image_size * image_size).tolist()
            
            memory.append((state, action, next_state, reward, is_done))

            if is_done:
                break

            estimator.replay(memory, replay_size, gamma)
            state = next_state

        print(f"Episode {episode}, total reward {total_reward_episode[episode]}, epsilon {epsilon}")
        epsilon = max(epsilon * epsilon_decay, 0.01)
    
    return total_reward_episode


if __name__ == '__main__':
    env = gym.envs.make("PongDeterministic-v4")
    
    image_size = 84
    n_state = image_size * image_size
    n_hidden = [200, 50]
    n_episode = 1000
    lr = 0.003
    replay_size = 32
    target_update = 10
    n_action = 3

    dqn = Double_DQN(n_state, n_action, n_hidden, lr)

    rewards = q_learning(env, dqn, n_episode, n_action, replay_size, target_update, image_size, gamma=0.9, epsilon=1.0, epsilon_decay=0.99)

    draw("pong_double_dqn_fc.png", "Rewards over time", "Episode", "Reward", rewards)