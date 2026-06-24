import logging
import torch

from afa.environments.environment import Environment
from afa.classifiers import ConfidentMultiFormer
from afa.datasets.echoprime_dataset import ef_mapping
from afa.utils import js_divergence

ACTION_2_VIDEO_IDX = {0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]}
ACTION_2_COST = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}


class EchoPrimeEnv(Environment):
    def __init__(self, data, n_actions: int, classifiers, lammy: float, max_step: int,
                 reward_scheme: str, task: str, device: str, logger: logging.Logger):
        super().__init__(n_actions)
        self.logger = logger
        self.device = device
        self.n_actions = n_actions
        self.classifiers = classifiers
        self.data = data
        self.lammy = lammy
        self.max_step = max_step
        self.reward_scheme = reward_scheme

        self.task = task
        if self.task in ['AS_EF', 'AS']:
            self.as_label = None
            self.as_pred = None
        if self.task in ['AS_EF', 'EF']:
            self.ef_cat_label = None
            self.ef_cat_pred = None
            self.ef_value_label = None
            self.ef_value_pred = None
            self.joint_pred_pmf = []

        self.current_patient = None
        self.super_state = None
        self.study_id = None
        self.current_state = None
        self.state_shape = None
        self.action_history = None

        self.reset(0)

    def n_episodes(self):
        return len(self.data)

    def reset(self, idx: int) -> (torch.Tensor, str):
        study = self.data[idx]
        with torch.no_grad():
            self.super_state = study['study_emb'].to(
                self.device)  # torch.tensor [5, 512], [AP2, AP3, AP4, PLAX, PSAXAo]
            self.as_label = study['as_severity'].item()
            self.ef_cat_label = study['ef_category'].item()
            self.ef_value_label = study['ef_value'].item()
            self.study_id = study['exam_id'].item()
            self.current_state = torch.zeros_like(self.super_state)
            self.state_shape = self.current_state.shape
            self.action_history = []

            if self.reward_scheme == "dense":
                self.update_pred_pmf(reset=True)

        return self.current_state.clone(), "Environment reset."

    def step(self, action: int) -> (torch.Tensor, float, bool, bool, str):
        assert type(action) == int, f"action type must be int, got type {type(action)}"
        assert action in self.action_space.range, f"action must be in action_space = {self.action_space.range}"

        action_is_available = True
        terminated = False
        reward = 0.0

        with (torch.no_grad()):
            if action == 0:
                terminated = True
                self.set_y_pred()
                if self.task in ['AS_EF', 'AS']:
                    # reward += 0
                    reward += int(self.as_pred == self.as_label) - self.lammy * self.get_cost()
                if self.task in ['AS_EF', 'EF']:
                    # reward += 0
                    reward += int(self.ef_cat_pred == self.ef_cat_label) - self.lammy * self.get_cost()
            else:
                if (self.super_state[ACTION_2_VIDEO_IDX[action]] == 0.5).all():
                    action_is_available = False
                else:
                    self.current_state[ACTION_2_VIDEO_IDX[action]] = self.super_state[ACTION_2_VIDEO_IDX[action]]

            if action_is_available:
                self.action_history.append(action)
                info = ""
                ### JS Divergence as Dense Reward
                if not terminated and self.reward_scheme == "dense":
                    self.update_pred_pmf()
                    reward += js_divergence(self.joint_pred_pmf[-2], self.joint_pred_pmf[-1]).item()
            else:
                for action in range(1, self.n_actions):
                    if (self.super_state[ACTION_2_VIDEO_IDX[action]] == 0.5).all() or action in self.action_history:
                        continue
                    self.current_state[ACTION_2_VIDEO_IDX[action]] = self.super_state[ACTION_2_VIDEO_IDX[action]]
                    self.action_history.append(action)
                terminated = True
                self.set_y_pred()
                if self.task in ['AS_EF', 'AS']:
                    reward += int(self.as_pred == self.as_label) - self.lammy * self.get_cost()
                if self.task in ['AS_EF', 'EF']:
                    reward += int(self.ef_cat_pred == self.ef_cat_label) - self.lammy * self.get_cost()
                self.action_history.append(0)
                info = "- took all available actions"

        truncated = len(self.action_history) == self.max_step
        if truncated and not terminated:
            self.set_y_pred()
            reward = -self.lammy * self.get_cost()

        return self.current_state.clone(), reward, terminated, truncated, info

    def get_cost(self) -> float:
        cost = 0
        for action in self.action_history:
            cost += ACTION_2_COST[action]

        return cost

    def set_y_pred(self):
        if isinstance(self.classifiers['EF'], ConfidentMultiFormer):
            out = self.classifiers['EF'].predict(self.current_state.clone(), self.action_history)
            self.ef_value_pred = out['reg_pred']
            self.ef_cat_pred = out['reg_pred_category']
            self.as_pred = out['class_pred']
            return
        if self.task in ['AS_EF', 'AS']:
            self.as_pred = self.classifiers['AS'].predict(self.current_state.clone(), self.action_history)['class_pred']
        if self.task in ['AS_EF', 'EF']:
            self.ef_value_pred = self.classifiers['EF'].predict(self.current_state.clone(), self.action_history)[
                'reg_pred']
            self.ef_cat_pred = ef_mapping(self.ef_value_pred)

    def update_pred_pmf(self, reset=False):
        pass
        assert self.task == 'AS_EF', f"Should be a multi-task multi-variate classifier"
        if reset:
            self.joint_pred_pmf = []
        self.joint_pred_pmf.append(
            self.classifiers['AS'].predict(self.current_state.clone(), self.action_history)['joint_probs'])

    def render(self):
        pass
