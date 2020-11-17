import torch
import random
import copy
from torch.nn import functional as F

class DQN():
    r"""Deep Q-Networks (DQN)
    
    Args:
        n_state: length of state (1st dimension)
        n_action: number of action
        n_hidden (int or list[int]): number of hidden units
        lr: learning rate
    """

    def __init__(self, n_state, n_action, n_hidden=50, lr=0.05):
        self.loss_func = torch.nn.MSELoss()
        if isinstance(n_hidden, int):
            self.model = torch.nn.Sequential(
                torch.nn.Linear(n_state, n_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(n_hidden, n_action)
            ).to(self.device)
        else:
            layer_list = []
            for i in range(len(n_hidden)):
                if i == 0:
                    layer_list.append(torch.nn.Linear(n_state, n_hidden[0]))
                    layer_list.append(torch.nn.ReLU())
                else:
                    layer_list.append(torch.nn.Linear(n_hidden[i-1], n_hidden[i]))
                    layer_list.append(torch.nn.ReLU())
                if i == len(n_hidden)-1:
                    layer_list.append(torch.nn.Linear(n_hidden[i], n_action))
            self.model = torch.nn.Sequential(*layer_list).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
    

    @property
    def device(self):
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'


    def update(self, state, target_value):
        r"""Update the weights of the DQN given a training example

        Args:
            state (list[list], list or tensor): the state, if it's a list, an extra dimension will be added at the first dimension
            target_value (list[list] or list): target value
        """
        if isinstance(state, list) and not isinstance(state[0], list):
            state = [state]
            target_value = [target_value]
        value_pred = self.model(torch.tensor(state, dtype=torch.float, device=self.device))
        loss = self.loss_func(value_pred, torch.tensor(target_value, dtype=torch.float, device=self.device))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    

    def predict(self, state):
        r"""Compute the Q values of the state for all actions using the learning model

        Args:
            state (list[list], list or tensor): the state, if it's a list, an extra dimension will be added at the first dimension
        Returns:
            Q values of the state for all actions, shape = (Batch_size * n_action), Batch_size = 1 by default
        """
        if isinstance(state, list) and not isinstance(state[0], list):
            state = [state]
        with torch.no_grad():
            return self.model(torch.tensor(state, dtype=torch.float, device=self.device)).cpu()
    

    def replay(self, memory, replay_size, gamma, model_target=None):
        """Experience replay

        Args:
            memory: a list of experience
            replay_size: the number of samples we use to update the model each time
            gamma: the discount factor
            model_target: default: None, the model for predicting target
        """
        if len(memory) >= replay_size:
            replay_data = random.sample(memory, replay_size)
            states = []
            targets = []

            for state, action, next_state, reward, is_done in replay_data:
                states.append(state)
                q_values = self.predict(state).tolist()[0]
                if is_done:
                    q_values[action] = reward
                else:
                    q_values_next = self.predict(next_state) if model_target is None else model_target(next_state)
                    q_values[action] = reward + gamma * torch.max(q_values_next).item()
                targets.append(q_values)
            
            self.update(states, targets)
    

class Double_DQN(DQN):
    """Double DQN, using a separate model for target prediction

    Args:
        n_state: length of state (1st dimension)
        n_action: number of action
        n_hidden: number of hidden units
        lr: learning rate
    """
    def __init__(self, n_state, n_action, n_hidden, lr):
        super().__init__(n_state, n_action, n_hidden, lr)
        self.model_target = copy.deepcopy(self.model)


    def target_predict(self, state):
        """Predict using the target model

        Args:
            state(list[list], list or tensor): the state, if it's list, an extra dimension will be added in the first dimension
        Returns:
            Q values of the state for all actions, shape = (Batch_size * n_action), Batch_size = 1 by default
        """
        if isinstance(state, list) and not isinstance(state[0], list):
            state = [state]
        with torch.no_grad():
            return self.model_target(torch.tensor(state, dtype=torch.float, device=self.device)).cpu()
    

    def copy_target(self):
        """Copy the weights from model to model_target"""
        self.model_target.load_state_dict(self.model.state_dict())
    

    def replay(self, memory, replay_size, gamma):
        """Experience replay

        Args:
            memory: a list of experience
            replay_size: the number of samples we use to update the model each time
            gamma: the discount factor
        """
        super().replay(memory, replay_size, gamma, model_target=self.target_predict)


class DuelingModel(torch.nn.Module):
    """Dueling DQN model

    Args:
        n_input: number of input (n_state)
        n_output: number of output (n_action)
        n_hidden: number of hidden units
    """
    def __init__(self, n_input, n_output, n_hidden):
        super().__init__()
        self.advantage = torch.nn.Sequential(
            torch.nn.Linear(n_input, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_output)
        ).to(self.device)
        self.value = torch.nn.Sequential(
            torch.nn.Linear(n_input, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, 1)
        ).to(self.device)
    

    def forward(self, x):
        adv = self.advantage(x)
        val = self.value(x)
        return val + adv - adv.mean()


class DDQN(DQN):
    """Dueling DQN

    Args:
        n_state: number of state
        n_action: number of action
        n_hidden: number of hidden units
        lr: learning rate
    """
    def __init__(self, n_state, n_action, n_hidden, lr):
        super().__init__(n_state, n_action, n_hidden, lr)
        self.model = DuelingModel(n_state, n_action, n_hidden)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)


class CNNModel(torch.nn.Module):
    """CNN model

    Args:
        n_channel: number of input channel
        n_action: number of action
    """
    def __init__(self, n_channel, n_action):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(n_channel, 32, 8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, 4, 2)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1)
        self.fc = torch.nn.Linear(7*7*64, 512)
        self.out = torch.nn.Linear(512, n_action)

    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        output = self.out(x)
        return output


class CNNDQN(Double_DQN):
    """CNN based Double DQN model

    Args:
        n_channel: number of input channel
        n_action: number of action
        lr: learning rate
    """
    def __init__(self, n_channel, n_action, lr=0.05):
        self.loss_func = torch.nn.MSELoss()
        self.model = CNNModel(n_channel, n_action).to(self.device)
        self.model_target = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
    

    def predict(self, state):
        """Predict using the model

        Args:
            state: tensor [1, 3, image_size, image_size]
        Returns:
            Q values: tensor [1, n_action]
        """
        with torch.no_grad():
            return self.model(state.to(self.device)).cpu()
    

    def target_predict(self, state):
        """Predict target using the target model

        Args:
            state: tensor, [1, 3, image_size, image_size]
        Returns:
            Q values: tensor [1, n_action]
        """
        with torch.no_grad():
            return self.model_target(state.to(self.device)).cpu()
    

    def replay(self, memory, replay_size, gamma):
        """Experience replay with target model

        Args:
            memory: a list of experience
            replay_size: the number of samples we use to update the model each time
            gamma: the discount factor
        """
        if len(memory) >= replay_size:
            replay_data = random.sample(memory, replay_size)
            states = []
            targets = []

            for state, action, next_state, reward, is_done in replay_data:
                states.append(state)
                q_values = self.predict(state)
                if is_done:
                    q_values[0][action] = reward
                else:
                    q_values_next = self.target_predict(next_state)
                    q_values[0][action] = reward + gamma * torch.max(q_values_next).item()
                targets.append(q_values)
            
            self.update(states, targets)
    
    
    def update(self, states, targets):
        """Update the prediction model using states and targets
        
        Args:
            states: list[tensor], tensor shape: [1, 3, image_size, image_size]
            targets: list[tensor], tensor shape: [1, n_action]
        """
        value_pred = self.model(torch.cat(states).to(self.device))
        loss = self.loss_func(value_pred, torch.cat(targets).to(self.device))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()