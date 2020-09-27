import matplotlib.pyplot as plt
import torch

def draw(file_name, title, x_label, y_label, y, x=None):
    plt.figure()
    if x is not None:
        plt.plot(x, y)
    else:
        plt.plot(y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(file_name)


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