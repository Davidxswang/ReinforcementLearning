import torch
import gym
from utils import draw
from dqn import DDQN
from mountain_car_dqn_experience_replay import q_learning


if __name__ == '__main__':
    env = gym.envs.make("MountainCar-v0")

    n_state = env.observation_space.shape[0]
    n_action = env.action_space.n
    n_hidden = 50
    lr = 0.001
    ddqn = DDQN(n_state, n_action, n_hidden, lr)

    n_episode = 600
    replay_size = 20
    rewards = q_learning(env, ddqn, n_episode, n_action, replay_size, gamma=0.9, epsilon=0.3, epsilon_decay=0.99)
    draw("mountain_car_dueling_dqn.png", "Rewards over episodes", "Episode", "Reward", rewards)