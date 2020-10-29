import gym
from neural_network_estimator import NeuralNetworkEstimator
from mountain_car_linear_estimator_q_learning_experience_replay import q_learning_with_replay
from utils import draw


if __name__ == '__main__':
    env = gym.envs.make('MountainCar-v0')
    
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.n
    n_feature = 200
    n_hidden = 50
    lr = 0.001

    estimator = NeuralNetworkEstimator(n_hidden, n_feature, n_state, n_action, lr)
    
    n_episode = 1000
    replay_size = 200

    total_reward_episode = q_learning_with_replay(env, estimator, n_episode, n_action, replay_size, gamma=1.0, epsilon=0.1, epsilon_decay=0.99)

    draw('mountain_car_neural_network_estimator_q_learning_experience_replay.png', 'Reward over episodes', 'episode', 'reward', total_reward_episode)