import torch
from base_estimator import Estimator


class NeuralNetworkEstimator(Estimator):
    def __init__(self, n_hidden, *args, **kwargs):
        self.n_hidden = n_hidden
        super().__init__(*args, **kwargs)
        


    def _init_models(self):
        for _ in range(self.n_action):
            model = torch.nn.Sequential(
                torch.nn.Linear(self.n_feature, self.n_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(self.n_hidden, 1)
            )
            model = model.to(self.device)
            self.models.append(model)
            optimizer = torch.optim.Adam(model.parameters(), self.lr)
            self.optimizers.append(optimizer)