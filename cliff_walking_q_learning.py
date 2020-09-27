import torch
import gym
from collections import defaultdict
import matplotlib.pyplot as plt
from utils import draw

env = gym.make('CliffWalking-v0')


def gen_epsilon_greedy_policy(n_action, epsilon):
    """Generate a epsilon greedy policy
    Args:
        n_action: the number of actions
        epsilon: epsilon
    Returns:
        the policy function: 
            inputs: state, Q
            output: action
    """
    def policy_function(state, Q):
        probs = torch.ones(n_action) * epsilon / n_action
        best_action = torch.argmax(Q[state]).item()
        probs[best_action] += 1 - epsilon
        action = torch.multinomial(probs, 1).item()
        return action
    return policy_function


def q_learning(env, gamma, n_episode, alpha, policy):
    """Obtain the optimal policy with off-policy Q-learning method
    Args:
        env: OpenAI Gym environment
        gamma: discount factor
        n_episode: number of episodes
        alpha: learning rate
        policy function: input: n_action, epsilon, output: action
    Returns:
        optimal Q-function, optimal policy, length of each episode, total reward for each episode
    """

    n_action = env.action_space.n
    Q = defaultdict(lambda: torch.zeros(n_action))
    length_episode = [0] * n_episode
    total_reward_episode = [0] * n_episode

    for episode in range(n_episode):
        state = env.reset()
        is_done = False
        while not is_done:
            action = policy(state, Q)
            next_state, reward, is_done, info = env.step(action)
            td_delta = reward + gamma * torch.max(Q[next_state]) - Q[state][action]
            Q[state][action] += alpha * td_delta
            length_episode[episode] += 1
            total_reward_episode[episode] += reward
            if is_done:
                break
            state = next_state
    policy = {}
    for state, actions in Q.items():
        policy[state] = torch.argmax(actions).item()
    return Q, policy, length_episode, total_reward_episode

if __name__ == '__main__':

    gamma = 1
    n_episode = 500
    alpha = 0.4
    epsilon = 0.1

    epsilon_greedy_policy = gen_epsilon_greedy_policy(env.action_space.n, epsilon)

    optimal_Q, optimal_policy, length, rewards = q_learning(env, gamma, n_episode, alpha, epsilon_greedy_policy)

    print(optimal_Q, optimal_policy)

    draw('cliff_walking_length_episodes.png', 'Episode length over time', 'Episode', 'Length', length)
    draw('cliff_walking_total_reward_episodes.png', 'Episode reward over time', 'Episode', 'Length', rewards)