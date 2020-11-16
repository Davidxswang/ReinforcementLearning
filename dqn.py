import torch
import random
import copy

class DQN():
    r"""Deep Q-Networks (DQN)
    
    Args:
        n_state: length of state (1st dimension)
        n_action: number of action
        n_hidden: number of hidden units
        lr: learning rate
    """

    def __init__(self, n_state, n_action, n_hidden=50, lr=0.05):
        self.loss_func = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_state, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_action)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
    

    def update(self, state, target_value):
        r"""Update the weights of the DQN given a training example

        Args:
            state (list[list] or list): state
            target_value (list[list] or list): target value
        """
        if not isinstance(state[0], list):
            state = [state]
            target_value = [target_value]
        value_pred = self.model(torch.tensor(state, dtype=torch.float))
        loss = self.loss_func(value_pred, torch.tensor(target_value, dtype=torch.float))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    

    def predict(self, state):
        r"""Compute the Q values of the state for all actions using the learning model

        Args:
            state (list[list] or list): state
        Returns:
            Q values of the state for all actions, shape = (Batch_size * n_action), Batch_size = 1 by default
        """
        if not isinstance(state[0], list):
            state = [state]
        with torch.no_grad():
            return self.model(torch.tensor(state, dtype=torch.float))
    

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_target = copy.deepcopy(self.model)


    def target_predict(self, state):
        """Predict using the target model

        Args:
            state(list[list] or list): the state
        Returns:
            Q values of the state for all actions, shape = (Batch_size * n_action), Batch_size = 1 by default
        """
        if not isinstance(state[0], list):
            state = [state]
        with torch.no_grad():
            return self.model_target(torch.tensor(state, dtype=torch.float))
    

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
        )
        self.value = torch.nn.Sequential(
            torch.nn.Linear(n_input, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, 1)
        )
    

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
