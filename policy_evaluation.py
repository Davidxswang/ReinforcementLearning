import torch
import matplotlib.pyplot as plt

def policy_evaluation(policy, trans_matrix, gamma, threshold, rewards):
    r"""Evaluate the value of a policy.

    Args:
        policy (Tensor, n_state*n_action): policy matrix containing actions and their probability in each state
        trans_matrix (Tensor, n_state*n_action*n_state): transformation matrix
        gamma (float): the discount factor controlling the tradeoff between current and future rewards
        threshold (float): the threshold to terminate the evaluation
        rewards (Tensor, n_state): rewards for each state
    Returns:
        value, values: the final value and the value history
    """

    n_state = rewards.shape[0]
    value = torch.zeros(n_state)
    values = [value]

    while True:
        value_temp = torch.zeros(n_state)
        for state, actions in enumerate(policy):
            # actions: Tensor, n_action
            for action, prob in enumerate(actions):
                value_temp[state] += prob * (rewards[state] + gamma * torch.dot(trans_matrix[state, action], value))
        max_delta = torch.max(torch.abs(value_temp - value))
        value = value_temp.clone()
        values.append(value)
        if max_delta <= threshold:
            break

    return value, values


def plot(value_history, gamma):
    plt.figure()
    state_0 = [v[0] for v in value_history]
    state_1 = [v[1] for v in value_history]
    state_2 = [v[2] for v in value_history]

    line_0, = plt.plot(state_0)
    line_1, = plt.plot(state_1)
    line_2, = plt.plot(state_2)

    plt.xlabel('iterations')
    plt.ylabel('value')
    plt.title('value function')
    plt.legend([line_0, line_1, line_2],
                ['state 0', 'state 1', 'state 2'],
                loc='upper left')

    plt.savefig(f'policy_evaluation_gamma_{gamma}.png')

if __name__ == '__main__':
    trans_matrix = torch.tensor(
        [
            [[0.8, 0.1, 0.1],
             [0.1, 0.6, 0.3]],
            [[0.7, 0.2, 0.1],
             [0.1, 0.8, 0.1]],
            [[0.6, 0.2, 0.2],
             [0.1, 0.4, 0.5]]
        ]
    )
    policy = torch.tensor(
        [
            [0.6, 0.4],
            [0.4, 0.6],
            [0.5, 0.5]
        ]
    )
    threshold = 0.0001
    gamma = 0.5
    rewards = torch.tensor([1., 0, -1.])
    
    for gamma in [0.2, 0.5, 0.9, 0.99]:
        value, value_history = policy_evaluation(policy, trans_matrix, gamma, threshold, rewards)
        plot(value_history, gamma)