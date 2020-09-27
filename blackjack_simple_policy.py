import torch
import gym
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

env = gym.make('Blackjack-v0')
# action 1: hit, 0: stick


def run_episode(env, hold_score):
    state = env.reset()
    rewards = []
    states = [state]
    is_done = False

    while not is_done:
        action = 1 if state[0] < hold_score else 0
        state, reward, is_done, info = env.step(action)
        states.append(state)
        rewards.append(reward)
        if is_done:
            break
    
    return states, rewards


def mc_prediction_first_visit(env, hold_score, gamma, n_episode):
    value = defaultdict(float)
    number = defaultdict(int)

    for episode in range(n_episode):
        states_t, rewards_t = run_episode(env, hold_score)
        return_t = 0
        G = {}
        
        for state_t, reward_t in zip(states_t[1::-1], rewards_t[::-1]):
            return_t = gamma * return_t + reward_t
            G[state_t] = return_t
        
        for state, return_t in G.items():
            if state[0] <= 21:
                value[state] += return_t
                number[state] += 1
    
    for state in value:
        value[state] /= number[state]

    return value


def plot_surface(x, y, z, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Player Sum')
    ax.set_ylabel('Dealer Showing')
    ax.set_zlabel('Value')
    ax.set_title(title)
    ax.view_init(ax.elev, -120)
    fig.colorbar(surf)
    fig.savefig(f'blackjack_value_{title.replace(" ", "_")}.png')


def plot_blackjack_value(value):
    player_sum_range = range(12, 22)
    dealer_sum_range = range(1, 11)
    x, y = torch.meshgrid([torch.tensor(player_sum_range), torch.tensor(dealer_sum_range)])
    values_to_plot = torch.zeros(len(player_sum_range), len(dealer_sum_range), 2)
    for i, player in enumerate(player_sum_range):
        for j, dealer in enumerate(dealer_sum_range):
            for k, ace in enumerate([False, True]):
                values_to_plot[i, j, k] = value[(player, dealer, ace)]
    
    plot_surface(x, y, values_to_plot[:, :, 0].numpy(), 'No Usable Ace')
    plot_surface(x, y, values_to_plot[:, :, 1].numpy(), 'With Usable Ace')


hold_score = 18
gamma = 1
n_episode = 500000

value = mc_prediction_first_visit(env, hold_score, gamma, n_episode)
plot_blackjack_value(value)