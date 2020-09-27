import gym
import torch
import matplotlib.pyplot as plt


def value_iteration(env, gamma, threshold):
    """
    Solve a given environment with value iteration algorithm.

    Args:
        env: OpenAI Gym environment
        gamma (float): discount factor
        threshold (float): will stop the iteration when all the changes <= threshold
    
    Returns:
        values for the optimal policy for the given environment
    """

    n_state = env.observation_space.n
    n_action = env.action_space.n

    value = torch.zeros(n_state)

    while True:
        value_temp = torch.empty(n_state)
        for state in range(n_state):
            v_actions = torch.zeros(n_action)
            for action in range(n_action):
                for trans_prob, new_state, reward, _ in env.env.P[state][action]:
                    v_actions[action] += trans_prob * (reward + gamma * value[new_state])
            value_temp[state] = torch.max(v_actions)
        max_delta = torch.max(torch.abs(value - value_temp))
        value = value_temp.clone()
        if max_delta <= threshold:
            break

    return value


def extract_optimal_policy(env, value_optimal, gamma):
    """
    Obtain the optimal policy based on the optimal values
    Args:
        env: OpenAI Gym environment
        value_optimal: the optimal value computed using value_iteration
        gamma: discount factor

    Returns:
        optimal policy
    """

    n_state = env.observation_space.n
    n_action = env.action_space.n
    optimal_policy = torch.zeros(n_state)

    for state in range(n_state):
        v_actions = torch.zeros(n_action)
        for action in range(n_action):
            for trans_prob, new_state, reward, _ in env.env.P[state][action]:
                v_actions[action] += trans_prob * (reward + gamma * value_optimal[new_state])
        optimal_policy[state] = torch.argmax(v_actions)
    
    return optimal_policy


def run_episode(env, policy):
    state = env.reset()
    total_reward = 0
    is_done = False
    while not is_done:
        action = policy[state].item()
        state, reward, is_done, _ = env.step(action)
        total_reward += reward
        if is_done:
            break
    return total_reward


def test_different_gammas(env, gammas, n_episode=10000, threshold=0.0001):
    """
    Run tests on different gammas over many episodes.
    Args:
        env: OpenAI Gym environment
        gammas (list): a list of gamms
        n_episode (int): how many episodes to run to get the result of avg reward
        threshold (float): control when to stop the value iteration
    Returns:
        Average reward over gammas
    """
    avg_reward_gamma = []
    for gamma in gammas:
        total_rewards = []
        value_optimal = value_iteration(env, gamma, threshold)
        optimal_policy = extract_optimal_policy(env, value_optimal, gamma)
        for episode in range(n_episode):
            total_reward = run_episode(env, optimal_policy)
            total_rewards.append(total_reward)
        avg_reward_gamma.append(sum(total_rewards) / n_episode)

    return avg_reward_gamma


def plot(x, y, title, xlabel, ylabel, savefig=False, figname=''):
    """
    A generic function to plot figure.
    Args:
        x (list): data for x-axis
        y (list): data for y-axis
        title (str): title string
        xlabel (str): label for x-axis
        ylabel (str): label for y-axis
        savefig (bool): whether to save the fig, if True, will save fig and not show the image
        figname (str): the figure name when saving the figure
    """
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if savefig:
        plt.savefig(figname)
    else:
        plt.show()


env = gym.make('FrozenLake-v0')
gammas = [0, 0.2, 0.4, 0.6, 0.8, 0.99, 1]

avg_reward_gamma = test_different_gammas(env, gammas)
plot(gammas, avg_reward_gamma, 'Success rate vs discount factor', 'discount factor', 'success rate', True, 'frozen_lake_value_gammas.png')