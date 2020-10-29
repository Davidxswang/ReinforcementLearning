import torch

from base_estimator import Estimator


class LinearEstimator(Estimator):
    def _init_models(self):
        for _ in range(self.n_action):
            model = torch.nn.Linear(self.n_feature, 1).to(self.device)
            self.models.append(model)
            optimizer = torch.optim.SGD(model.parameters(), self.lr)
            self.optimizers.append(optimizer)