import torch
import gym
import random


if __name__ == '__main__':
    env = gym.envs.make("PongDeterministic-v4")

    state_shape = env.observation_space.shape
    n_action = env.action_space.n
    print(state_shape)
    print(n_action)
    print(env.unwrapped.get_action_meanings())

    ACTIONS = [0, 2, 3]
    n_action = 3

    env.reset()
    is_done = False

    while not is_done:
        action = ACTIONS[random.randint(0, n_action-1)]
        state, reward, is_done, _ = env.step(action)
        print(reward, is_done)
        env.render()