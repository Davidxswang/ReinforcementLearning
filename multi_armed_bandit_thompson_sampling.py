import torch
from multi_armed_bandit import BanditEnv
from utils import thompson_sampling


if __name__ == '__main__':
    bandit_payout = [0.01, 0.015, 0.03]
    bandit_reward = [1, 1, 1]
    bandit_env = BanditEnv(bandit_payout, bandit_reward)

    n_episode = 100000
    n_action = len(bandit_payout)
    action_count = torch.tensor([0. for _ in range(n_action)])
    action_total_reward = [0 for _ in range(n_action)]
    action_avg_reward = [[] for action in range(n_action)]

    
    alpha = torch.ones(n_action)
    beta = torch.ones(n_action)

    for episode in range(n_episode):
        action = thompson_sampling(alpha, beta)
        reward = bandit_env.step(action)
        action_count[action] += 1
        action_total_reward[action] += reward
        if reward > 0:
            alpha[action] += 1
        else:
            beta[action] += 1


        for a in range(n_action):
            if action_count[a]:
                action_avg_reward[a].append(action_total_reward[a] / action_count[a])
            else:
                action_avg_reward[a].append(0)
        
    import matplotlib.pyplot as plt
    for action in range(n_action):
        plt.plot(action_avg_reward[action])
    plt.legend(['Arm {}'.format(action) for action in range(n_action)])
    plt.title('Average reward over time')
    plt.xscale('log')
    plt.xlabel('Episode')
    plt.ylabel('Average reward')
    plt.savefig('multi_armed_bandit_thompson_sampling.png')