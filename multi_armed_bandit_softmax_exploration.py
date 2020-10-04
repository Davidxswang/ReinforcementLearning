import torch
from multi_armed_bandit import BanditEnv
from utils import gen_softmax_exploration_policy


if __name__ == '__main__':
    bandit_payout = [0.1, 0.15, 0.3]
    bandit_reward = [4, 3, 1]
    bandit_env = BanditEnv(bandit_payout, bandit_reward)

    n_episode = 100000
    n_action = len(bandit_payout)
    action_count = [0 for _ in range(n_action)]
    action_total_reward = [0 for _ in range(n_action)]
    action_avg_reward = [[] for action in range(n_action)]

    tau = 0.1
    softmax_exploration_policy = gen_softmax_exploration_policy(tau)
    Q = torch.zeros(1, n_action)

    for episode in range(n_episode):
        action = softmax_exploration_policy(0, Q)
        reward = bandit_env.step(action)
        action_count[action] += 1
        action_total_reward[action] += reward
        Q[0, action] = action_total_reward[action] / action_count[action]

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
    plt.savefig('multi_armed_bandit_softmax_exploration.png')