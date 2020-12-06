import torch
from torch import nn
from torch.nn import functional as F
from dqn import DQN


class PolicyNetwork(DQN):
    """A policy network

    Args:
        n_state: number of state
        n_action: number of action
        n_hidden: number of hidden units
        lr: learning rate
    """
    def __init__(self, n_state, n_action, n_hidden=50, lr=0.001):
        self.model = nn.Sequential(
            nn.Linear(n_state, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_action),
            nn.Softmax()
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
    

    def predict(self, state):
        """Predict the action probabilities using the model

        Args:
            state: input state
        Returns:
            predicted policy
        """
        return self.model(torch.Tensor(state).to(self.device))
    

    def update(self, returns, log_probs):
        """Update the weights of the policy network given the training examples

        Args:
            returns: if without baseline, it's cumulative rewards for each step in an episode; if with baseline, it's advantages for each step in an episode.
            log_probs: log probability for each step
        """
        policy_gradient = []
        for log_prob, Gt in zip(log_probs, returns):
            # log_prob is on cuda device
            # Gt is on cpu device
            # but they can multiply
            # because they are both 0-dimension tensor
            # cpu\cuda  0-D     n-D
            # 0-D       cuda    cuda
            # n-D       error   error
            # conclusion: if cpu tensor is a 0-D tensor, it can multiply with a cuda tensor, no matter what dimension the cuda tensor has
            policy_gradient.append(-log_prob * Gt.to(self.device))
        
        loss = torch.stack(policy_gradient).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    

    def get_action(self, state):
        """Estimate the policy and sample an action, compute its log probability
        
        Args:
            state: input state
        Returns:
            the selected action (int) and its log probability (tensor)
        """
        probs = self.predict(state)
        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs[action])
        return action, log_prob


class ValueNetwork(DQN):
    """Fully-connected network to calculate the value of the state

    Args:
        n_state: number of state
        n_hidden: number of hidden units
        lr: learning rate
    """
    def __init__(self, n_state, n_hidden=50, lr=0.05):
        self.loss_func = nn.MSELoss()
        self.model = torch.nn.Sequential(
            nn.Linear(n_state, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1)
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
    

    def update(self, state, value):
        """Update the model with state and target value

        Args:
            state: the input state
            value: the target value
        """
        y_pred = self.model(torch.Tensor(state).to(self.device))
        loss = self.loss_func(y_pred, torch.Tensor(value).to(self.device).unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    

    def predict(self, state):
        """Predict the value under certain state

        Args:
            state: the input state
        Returns:
            the value of the state
        """
        with torch.no_grad():
            return self.model(torch.Tensor(state).to(self.device)).cpu()


class ActorCriticModel(nn.Module):
    """Actor-critic algorithm model

    Args:
        n_input: number of input
        n_output: number of output
        n_hidden: number of hidden units, either an int, or List[int]
    Returns:
        action probabilities under the state, values of the state
    """
    def __init__(self, n_input, n_output, n_hidden):
        super().__init__()
        if isinstance(n_hidden, int):
            n_hidden = [n_hidden]
        
        fc_list = [nn.Linear(n_input, n_hidden[0]), nn.ReLU()]
        if len(n_hidden) > 1:
            for i in range(len(n_hidden)-1):
                fc_list.append(nn.Linear(n_hidden[i], n_hidden[i+1]))
                fc_list.append(nn.ReLU())

        self.backbone = nn.Sequential(*fc_list)
        self.action = nn.Linear(n_hidden[-1], n_output)
        self.value = nn.Linear(n_hidden[-1], 1)

    
    def forward(self, x):
        x = self.backbone(x)
        action_probs = F.softmax(self.action(x), dim=-1)
        state_values = self.value(x)
        return action_probs, state_values
    

class ActorCriticPolicyNetwork(PolicyNetwork):
    """Policy Network with Actor-Critic algorithm

    Args:
        n_state: length of state
        n_action: number of action
        n_hidden: number of hidden units
        lr: learning rate
    """
    def __init__(self, n_state, n_action, n_hidden=50, lr=0.001):
        self.model = ActorCriticModel(n_state, n_action, n_hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)
    

    def predict(self, state):
        """Predict the action probabilities of action under the input state

        Args:
            state: the input state
        Returns:
            the action probabilities, state-value
        """
        return self.model(torch.Tensor(state).to(self.device))
    

    def update(self, returns, log_probs, state_values):
        """Update the weights of the Actor Critic network given the training samples

        Args:
            returns: return (cumulative rewards) for each step in an episode
            log_probs: log probability for each step
            state_values: state-value for each step
        """
        loss = 0

        returns = returns.to(self.device).view(-1, 1)
        for log_prob, value, Gt in zip(log_probs, state_values, returns):
            advantage = Gt - value.item()
            policy_loss = -log_prob * advantage
            value_loss = F.smooth_l1_loss(value, Gt)
            loss += policy_loss + value_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def get_action(self, state):
        """Estimate the policy and sample an action, compute its log probability

        Args:
            state: input state
        Returns:
            the selected action, its log probability, and its state-value
        """
        action_probs, state_value = self.predict(state)
        action = torch.multinomial(action_probs, 1).item()
        log_prob = torch.log(action_probs[action])
        return action, log_prob, state_value


class ActorCriticGuassianModel(nn.Module):
    """Using a Guassian Model to simulate a Gaussian distribution and calculate the state-value

    Args:
        n_input: number of input
        n_output: number of output
        n_hidden: number of hidden units
    """
    def __init__(self, n_input, n_output, n_hidden):
        super().__init__()
        self.fc = nn.Linear(n_input, n_hidden)
        self.mu = nn.Linear(n_hidden, n_output)
        self.sigma = nn.Linear(n_hidden, n_output)
        self.value = nn.Linear(n_hidden, 1)
        self.distribution = torch.distributions.normal.Normal
    

    def forward(self, x):
        x = F.relu(self.fc(x))
        mu = 2 * torch.tanh(self.mu(x))
        sigma = F.softplus(self.sigma(x)) + 1e-5
        dist = self.distribution(mu.view(1, ).detach(), sigma.view(1, ).detach())
        value = self.value(x)
        return dist, value


class ActorCriticGaussianPolicyNetwork(ActorCriticPolicyNetwork):
    """Actor-critic algorithm using a Gaussian estimation network

    Args:
        n_state: number of state
        n_action: number of action
        n_hidden: number of hidden units
        lr: learning rate
    """
    def __init__(self, n_state, n_action, n_hidden=50, lr=0.001):
        self.model = ActorCriticGuassianModel(n_state, n_action, n_hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
    

    def predict(self, state):
        """Compute the distribution and state-value based on the input state

        Args:
            state: the input state
        Returns:
            dist (Normal object), value
        """
        #self.model.training = False
        result = self.model(torch.Tensor(state).to(self.device))
        #self.model.training = True
        return result
    

    def get_action(self, state):
        """Compute the action based on the state

        Args:
            state: the input state
        Returns:
            action, its log probability, and its estimated state-value
        """
        dist, value = self.predict(state)
        action = dist.sample().cpu().numpy()
        log_prob = dist.log_prob(action[0])
        return action, log_prob, value


class Estimator(PolicyNetwork):
    """Estimator network that can predict the action directly from the state

    Args:
        n_state: number of state
        lr: learning rate
    """
    def __init__(self, n_state, lr=0.0001):
        self.model = nn.Sequential(
            nn.Linear(n_state, 1),
            nn.Sigmoid()
        ).to(self.device)
        self.loss_func = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
    

    def predict(self, state):
        """Predict the action using the input state

        Args:
            state: the input state
        Returns:
            the action
        """
        return self.model(torch.Tensor(state).to(self.device))

    
    def update(self, state, target):
        """Update the model using the state and the target

        Args:
            state: the input state
            target: the target action
        """
        y_pred = self.predict(state)
        loss = self.loss_func(y_pred, torch.Tensor(target).to(self.device).view(-1, 1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()