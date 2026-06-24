import os
import logging
import torch
import torch.nn as nn
from torch.distributions import Categorical


class PPOAgent:
    def __init__(self, n_actions: int, discount_factor: float, device: str, logger: logging.Logger,
                 update_n_epochs: int, clip_epsilon: float = 0.2, c1: float = 1.0, c2: float = 0.01,
                 gae_lambda: float = 0.95):
        """
        Initialize PPO Agent
        Args:
            n_actions: Number of possible actions
            discount_factor: Discount factor for future rewards
            device: Device to run computations on
            logger: Logger instance
            clip_epsilon: PPO clipping parameter
            c1: Value function coefficient
            c2: Entropy coefficient
        """
        self.n_actions = n_actions
        self.device = device
        self.logger = logger
        self.discount_factor = discount_factor

        # PPO specific parameters
        self.clip_epsilon = clip_epsilon
        self.c1 = c1  # Value function coefficient
        self.c2 = c2  # Entropy coefficient
        self.gae_lambda = gae_lambda
        self.update_n_epochs = update_n_epochs

        self.actor = None
        self.critic = None
        self.actor_optimizer = None
        self.critic_optimizer = None

        # Initialize memory buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []

        self.checkpoint_epoch = None

    def take_one_action(self, state: torch.Tensor, phase: str) -> int:
        """
        Select an action using the current policy
        """
        with torch.no_grad():
            action_probs = self.actor(state.unsqueeze(0))
            if phase == 'train':
                dist = Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                self.log_probs.append(log_prob)
            else:
                action = torch.argmax(action_probs)

            return action.item()

    def update(self) -> float:
        """Update policy (actor) and value (critic) networks using PPO"""
        with torch.no_grad():
            states = torch.stack(self.states)
            actions = torch.tensor(self.actions, dtype=torch.long, device=self.device)
            rewards = torch.tensor(self.rewards, dtype=torch.float, device=self.device)
            next_states = torch.stack(self.next_states)
            dones = torch.tensor(self.dones, dtype=torch.float, device=self.device)
            old_log_probs = torch.stack(self.log_probs).squeeze(-1)

            # clear memory
            self.clear_memory()

            # Compute returns and advantages
            values = self.critic(states).squeeze(-1)
            next_values = self.critic(next_states).squeeze(-1)
            advantages = self.compute_advantages(rewards, values, next_values, dones)
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update for multiple epochs
        loss_sum = 0
        for _ in range(self.update_n_epochs):
            # Get current action probabilities
            action_probs = self.actor(states)
            dist = Categorical(action_probs)
            curr_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # Compute ratio and surrogate losses
            ratios = torch.exp(curr_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

            # Compute actor and critic losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(self.critic(states), returns)

            # Combined loss with entropy bonus
            loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy

            # Update actor and critic
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            loss_sum += loss.item()

        return loss_sum / self.update_n_epochs

    def compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor,
                           dones: torch.Tensor):
        with torch.no_grad():
            # Calculate TD errors (deltas)
            deltas = rewards + self.discount_factor * next_values * (1 - dones) - values

            # Initialize advantages tensor and the last advantage
            advantages = torch.zeros_like(deltas)
            gae = torch.zeros(1, device=rewards.device)

            # Work backwards to compute advantages
            for t in reversed(range(len(deltas))):
                gae = deltas[t] + self.discount_factor * self.gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae

            return advantages

    def store_transition(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool):
        """Store a transition in memory"""
        assert not state.requires_grad, "State should not require gradients."
        assert not next_state.requires_grad, "Next_state should not require gradients."

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []

    def save(self, save_dir, epoch, name='RL_agent'):
        torch.save({
            'epoch': epoch,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, os.path.join(save_dir, f'{name}.pth'))

    def load(self, checkpoint_path):
        loaded_checkpoint = torch.load(checkpoint_path)
        self.checkpoint_epoch = loaded_checkpoint['epoch']
        self.actor.load_state_dict(loaded_checkpoint['actor_state_dict'])
        self.critic.load_state_dict(loaded_checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(loaded_checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(loaded_checkpoint['critic_optimizer_state_dict'])
