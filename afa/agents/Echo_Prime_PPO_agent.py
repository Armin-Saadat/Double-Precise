import logging
import torch.optim as optim
import torch.nn as nn
from afa.agents.PPO_agent import PPOAgent


class EchoPrimePPOAgent(PPOAgent):

    def __init__(self, state_shape: tuple, n_actions: int, lr: float, milestones: list, gamma: float,
                 discount_factor: float, device: str, logger: logging.Logger, update_n_epochs: int):
        super().__init__(n_actions, discount_factor, device, logger, update_n_epochs)

        self.actor = ActorNetwork(state_shape, self.n_actions).to(device)
        self.critic = CriticNetwork(state_shape, self.n_actions).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.actor_optimizer, milestones=milestones,
                                                        gamma=gamma)


class ActorNetwork(nn.Module):
    def __init__(self, input_dim: tuple, n_actions: int):
        super().__init__()

        h = 1
        for i in input_dim:
            h = h * i

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(h, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.network(x)


class CriticNetwork(nn.Module):
    def __init__(self, input_dim: tuple, n_actions: int):
        super().__init__()

        h = 1
        for i in input_dim:
            h = h * i

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(h, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.network(x)
