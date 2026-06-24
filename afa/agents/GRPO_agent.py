import os
import logging
import torch
from torch.distributions import Categorical


class GRPOAgent:
    def __init__(self, n_actions: int, discount_factor: float, device: str, logger: logging.Logger,
                 update_n_epochs: int, group_size: int = 8, clip_epsilon: float = 0.2, c2: float = 0.01):
        """
        Initialize GRPO (Group Relative Policy Optimization) Agent
        Args:
            n_actions: Number of possible actions
            discount_factor: Discount factor for future rewards
            device: Device to run computations on
            logger: Logger instance
            update_n_epochs: Number of update epochs
            clip_epsilon: GRPO clipping parameter
            c2: Entropy coefficient
            group_size: Size of policy groups for GRPO
        """
        self.n_actions = n_actions
        self.device = device
        self.logger = logger
        self.discount_factor = discount_factor

        # GRPO specific parameters
        self.clip_epsilon = clip_epsilon
        self.c2 = c2  # Entropy coefficient
        self.update_n_epochs = update_n_epochs
        self.group_size = group_size  # Number of policies in a group

        self.actor = None
        self.actor_optimizer = None

        # Initialize memory buffers
        self.states = []
        self.actions = []
        self.rewards = []
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
        """Update policy network (actor) using GRPO"""
        with torch.no_grad():
            states = torch.stack(self.states)
            actions = torch.tensor(self.actions, dtype=torch.long, device=self.device)
            rewards = torch.tensor(self.rewards, dtype=torch.float, device=self.device)
            dones = torch.tensor(self.dones, dtype=torch.float, device=self.device)
            old_log_probs = torch.stack(self.log_probs).squeeze(-1)

            # Clear memory
            self.clear_memory()

            # Compute returns
            returns = self.compute_returns(rewards, dones)

        # GRPO update for multiple epochs
        batch_size, state_dim = states.shape[0], states.shape[1:]
        assert batch_size % self.group_size == 0, "Batch size must be a multiple of group_size"
        loss_sum = 0
        for _ in range(self.update_n_epochs):
            # Shuffle all data for this epoch
            indices = torch.randperm(batch_size)
            states = states[indices]
            actions = actions[indices]
            returns = returns[indices]
            old_log_probs = old_log_probs[indices]

            # Calculate group-normalized returns
            grouped_returns = returns.view(-1, self.group_size)
            group_means = grouped_returns.mean(dim=1, keepdim=True)
            group_stds = grouped_returns.std(dim=1, keepdim=True) + 1e-8
            normalized_grouped_returns = (grouped_returns - group_means) / group_stds
            normalized_returns = normalized_grouped_returns.reshape(-1)

            # Get current action probabilities
            action_probs = self.actor(states)
            dist = Categorical(action_probs)
            curr_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # Compute ratio and GRPO surrogate objective
            ratios = torch.exp(curr_log_probs - old_log_probs)
            surr1 = ratios * normalized_returns
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * normalized_returns

            # actor loss
            actor_loss = -torch.min(surr1, surr2).mean()

            # Add entropy bonus to encourage exploration
            loss = actor_loss - self.c2 * entropy

            # Update the actor network
            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            loss_sum += loss.item()

        return loss_sum / self.update_n_epochs

    def compute_returns(self, rewards: torch.Tensor, dones: torch.Tensor):
        """
        Compute discounted returns for each step
        """
        with torch.no_grad():
            returns = torch.zeros_like(rewards)
            running_return = 0.0

            # Compute discounted returns
            for t in reversed(range(len(rewards))):
                running_return = rewards[t] + self.discount_factor * running_return * (1 - dones[t])
                returns[t] = running_return

            return returns

    def store_transition(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool):
        """Store a transition in memory"""
        assert not state.requires_grad, "State should not require gradients."
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []

    def save(self, save_dir, epoch, mode='latest'):
        torch.save({
            'epoch': epoch,
            'actor_state_dict': self.actor.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
        }, os.path.join(save_dir, f'{mode}_RL_model.pth'))

    def load(self, save_dir, mode='latest'):
        loaded_checkpoint = torch.load(os.path.join(save_dir, f'{mode}_RL_model.pth'))
        self.checkpoint_epoch = loaded_checkpoint['epoch']
        self.actor.load_state_dict(loaded_checkpoint['actor_state_dict'])
        self.actor_optimizer.load_state_dict(loaded_checkpoint['actor_optimizer_state_dict'])
