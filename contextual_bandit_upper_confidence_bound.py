import torch
from utils import upper_confidence_bound
from multi_armed_bandit import BanditEnv


if __name__ == '__main__':
    bandit_payout_machines = [
        [0.01, 0.015, 0.03],
        [0.025, 0.01, 0.015]
    ]
    bandit_reward_machines = [
        [1, 1, 1],
        [1, 1, 1]
    ]
    
    n_machines = len(bandit_payout_machines)
    bandit_env_machines = [BanditEnv(bandit_payout, bandit_reward) for bandit_payout, bandit_reward in zip(bandit_payout_machines, bandit_reward_machines)]

    n_episode = 100000
    n_action = len(bandit_payout_machines[0])
    action_count = torch.zeros(n_machines, n_action)
    action_total_reward = torch.zeros(n_machines, n_action)
    action_avg_reward = [[[] for action in range(n_action)] for _ in range(n_machines)]

    Q = torch.zeros(n_machines, n_action)

    for episode in range(n_episode):
        state = torch.randint(0, n_machines, (1,)).item()
        action = upper_confidence_bound(Q, state, action_count[state], episode)
        reward = bandit_env_machines[state].step(action)
        action_count[state][action] += 1
        action_total_reward[state][action] += reward
        Q[state, action] = action_total_reward[state][action] / action_count[state][action]

        for a in range(n_action):
            if action_count[state][a]:
                action_avg_reward[state][a].append(action_total_reward[state][a] / action_count[state][a])
            else:
                action_avg_reward[state][a].append(0)
        
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 20))
    for state in range(n_machines):
        plt.subplot(n_machines, 1, state+1)
        for action in range(n_action):
            plt.plot(action_avg_reward[state][action])
        plt.legend(['Arm {}'.format(action) for action in range(n_action)])
        plt.title('Average reward over time for state {}'.format(state))
        plt.xscale('log')
        plt.xlabel('Episode')
        plt.ylabel('Average reward')
    plt.savefig('contextual_bandit_upper_confidence_bound.png')