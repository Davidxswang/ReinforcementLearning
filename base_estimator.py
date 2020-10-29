import torch
import math

class Estimator():
    r"""Estimator Base Class

    Subclass needs to implement the _init_models method
    """
    def __init__(self, n_feature, n_state, n_action, lr=0.05):
        r"""
        Args:
            n_feature: number of features
            n_state: number of states
            n_action: number of actions
            lr: learning rate
        """
        self.weight, self.bias = self.get_gaussian_wb(n_feature, n_state)
        self.n_feature = n_feature
        self.n_state = n_state
        self.n_action = n_action
        self.lr = lr
        self.models = []
        self.optimizers = []
        self.loss = torch.nn.MSELoss()
        self._init_models()
    

    def _init_models(self):
        raise NotImplementedError()


    @property
    def device(self):
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'


    def get_gaussian_wb(self, n_feature, n_state, sigma=0.2):
        r"""
        Generate the coeefficients of the feature set from Gaussian distribution

        Args:
            n_feature: number of features
            n_state: number of states
            sigma: standard deviation of the weight
        Returns:
            a tuple: weight, bias
        """
        torch.manual_seed(0)
        weight = torch.randn(n_state, n_feature) * 1.0 / sigma
        bias = torch.randn(n_feature) * 2.0 * math.pi
        weight.requires_grad_()
        bias.requires_grad_()
        weight = weight.to(self.device)
        bias = bias.to(self.device)
        return weight, bias


    def get_features(self, state):
        r"""
        Generate the features based on the input state

        Args:
            state: the input state
        Returns:
            the features        
        """
        features = math.sqrt(2.0 / self.n_feature) * torch.cos(torch.matmul(torch.tensor([state], device=self.device).float(), self.weight) + self.bias)
        return features

    
    def update(self, state, action, target):
        r"""
        Update the weights for the linear estimator with the given training sample

        Args:
            state: input state
            action: action
            target: the target value
        """
        features = self.get_features(state)
        pred = self.models[action](features).view(-1)
        loss = self.loss(pred, torch.tensor([target], device=self.device))
        self.optimizers[action].zero_grad()
        loss.backward()
        self.optimizers[action].step()


    def predict(self, state):
        r"""
        Compute the Q values of the state using the learning model
        
        Args:
            state: the input state
        Returns:
            a tensor containing the Q values of every action in the given state
        """
        features = self.get_features(state)
        with torch.no_grad():
            return torch.tensor([model(features) for model in self.models]).cpu()