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


def gen_softmax_exploration_policy(tau):
    """Generate a softmax exploration policy
    Args:
        tau (float): to control the exploration and exploitation, -> 0, towards best action, -> 1, towards equal exploration
    Returns:
        the policy function:
            inputs: state, Q
            output: action
    """
    def policy_function(state, Q):
        probs = torch.exp(Q[state] / tau)
        probs = probs / torch.sum(probs)
        action = torch.multinomial(probs, 1).item()
        return action
    return policy_function


def upper_confidence_bound(Q, state, action_count, episode):
    """
    Return an action with the highes upper confidence bound
    Args:
        Q (torch.Tensor): Q-function
        state (int): the state the env is on
        action_count (torch.Tensor): how many times each action has appeared, should be a FloatTensor
        episode (int): the number of episode the algorithm is currently on
    Returns:
        the best action (int)
    """
    ucb = torch.sqrt(2 * torch.log(torch.tensor(float(episode))) / action_count) + Q[state]
    return torch.argmax(ucb)


def thompson_sampling(alpha, beta):
    """
    Return an action based on beta distribution
    Args:
        alpha: alpha in beta distribution
        beta: beta in beta distribution
    Returns:
        action (int): the best action

    Note: each beta distribution should start with alpha=beta=1
    """
    prior_values = torch.distributions.beta.Beta(alpha, beta).sample()
    return torch.argmax(prior_values)