import logging
import os
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim

from afa.utils import powerset

# action 0 -> terminate the episode
ACTION_2_STATE_IDX = {0: [], 1: [0, 1], 2: [2, 3, 4], 3: [5, 6, 7, 8], 4: [9, 10]}


class UCIFullyConnected:
    def __init__(self, scaler: str, lr: float, n_epochs: int, mask_p: float, logger: logging.Logger):
        self.lr = lr
        self.n_epochs = n_epochs
        self.mask_p = mask_p

        self.model = None
        self.logger = logger

        if scaler == 'StandardScaler':
            self.scaler = StandardScaler()
        elif scaler == 'MinMaxScaler':
            self.scaler = MinMaxScaler()
        elif scaler == 'None':
            self.scaler = None
        else:
            raise ValueError(
                f"{scaler} is undefined. Here are the options for scaler: None, StandardScaler, MinMaxScaler")

    def train(self, data: pd.DataFrame):
        X = data[
            ['age', 'sex', 'cp', 'trestbps', 'restecg', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'slope']].values
        y = data['label'].values

        if self.scaler:
            X = self.scaler.fit_transform(X)

        X, y = torch.from_numpy(X).to(torch.float32), torch.from_numpy(y).to(torch.long)

        self.model = FCN(h=X.shape[1]*2, n_classes=2)
        self.model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.n_epochs):
            # mask = torch.bernoulli(torch.full(X.shape, self.mask_p))

            mask = torch.bernoulli(torch.full((X.shape[0], 4), self.mask_p))
            g1 = mask[:, 0].unsqueeze(1)
            g2 = mask[:, 1].unsqueeze(1)
            g3 = mask[:, 2].unsqueeze(1)
            g4 = mask[:, 3].unsqueeze(1)
            mask = torch.cat((g1, g1, g2, g2, g2, g3, g3, g3, g3, g4, g4), dim=1)

            X_masked = X * mask
            y_pred = self.model(torch.cat((X_masked, mask), dim=1))
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.model.eval()
        self.validate(data, phase='train')

    def validate(self, data: pd.DataFrame, phase='validation'):
        X = data[
            ['age', 'sex', 'cp', 'trestbps', 'restecg', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'slope']].values
        y = data['label'].values

        if self.scaler:
            X = self.scaler.transform(X)

        X, y = torch.from_numpy(X).to(torch.float32), torch.from_numpy(y).to(torch.long)

        accuracies = []
        feature_groups = [1, 2, 3, 4]
        for subset in powerset(feature_groups):
            mask = torch.zeros(X.shape, dtype=torch.float32)
            for a in subset:
                mask[:, ACTION_2_STATE_IDX[a]] = 1.0
            X_masked = X * mask
            with torch.no_grad():
                y_pred = self.model(torch.cat((X_masked, mask), dim=1))
                _, y_pred_class = torch.max(y_pred, 1)
            accuracies.append({'subset': subset, 'accuracy': accuracy_score(y, y_pred_class)})

        sorted_accuracies = sorted(accuracies, key=lambda x: x['accuracy'])
        self.logger.warning(f'Classifier performance on {phase} data:')
        for t in reversed(sorted_accuracies):
            self.logger.info('Feature subgroups {} --> Accuracy: {:.3f}'.format(t['subset'], t['accuracy']))

    def predict(self, X, action_history):

        # predict 0 when no data is acquired
        if not action_history:
            return 0

        actions_sorted = tuple(sorted(set(action_history)))
        mask = torch.zeros(X.size, dtype=torch.float32)
        for a in actions_sorted:
            mask[ACTION_2_STATE_IDX[a]] = 1.0
        X_masked = torch.from_numpy(X).to(torch.float32) * mask
        with torch.no_grad():
            y_pred = self.model(torch.cat((X_masked.unsqueeze(0), mask.unsqueeze(0)), dim=1))
            _, y_pred_class = torch.max(y_pred, 1)

        return bool(y_pred_class.item())

    def save(self, save_dir):
        with open(os.path.join(save_dir, 'fcn.pkl'), 'wb') as file:
            pickle.dump(self.model, file)

    def load(self, path):
        with open(path, 'wb') as file:
            self.model = pickle.load(file)


class FCN(nn.Module):
    def __init__(self, h: int, n_classes: int):
        super(FCN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(h, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes)
        )

    def forward(self, x):
        return self.fc(x)
