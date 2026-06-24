import os
import random
import logging
import wandb
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error

from afa.builders import data_builder, classifier_builder, agent_builder, environment_builder
from afa.utils import get_classification_metrics, Profiler
from afa.agents import DDQNAgent, PPOAgent, GRPOAgent


class Engine(object):
    def __init__(self, config, logger: logging.Logger, save_dir: str):
        self.logger = logger
        self.save_dir = save_dir
        self.config = config
        if self.config.device == 'cuda':
            self.logger.warning('Using {} GPU/s!'.format(torch.cuda.device_count()))
        else:
            self.logger.warning(f'Using {self.config.device}!')

        # Set up Wandb if required
        if config.use_wandb:
            wandb.init(project=config.wandb_project_name,
                       name=self.save_dir.removeprefix('./logs/'),
                       config=config,
                       mode=config.wandb_mode)

        # set data
        self.config.data.update({'task': self.config.task})
        self.data_train, self.data_val, self.data_test = data_builder.build(self.config.data)

        # set classifier
        self.config.classifier.update({'task': self.config.task, 'device': self.config.device})
        self.classifier = classifier_builder.build(self.config.classifier, self.logger)

        # set profilers
        self.profilers = {'Classifier Predictions': Profiler('Classifier Predictions', self.logger),
                          'Agent Taking Actions': Profiler('Agent Taking Actions', self.logger),
                          'Agent Updates': Profiler('Agent Updates', self.logger),
                          'Agent Store Transitions': Profiler('Agent Store Transitions', self.logger)}

        self.n_transitions = 0
        self.env = None
        self.agent = None

    def train(self):
        # set env
        self.config.environment.update({'task': self.config.task, 'device': self.config.device})
        self.env = environment_builder.build(self.config.environment, self.logger, self.classifier, self.data_train)

        # set agent
        self.config.agent.update({
            'state_shape': self.env.state_shape,
            'n_actions': self.env.action_space.n,
            'device': self.config.device})
        self.agent = agent_builder.build(self.config.agent, self.logger)

        max_reward = float('-inf')
        for epoch in range(self.config.n_epochs):

            ######################### train one epoch #########################
            self.env.data = self.data_train
            with Profiler() as p:
                metrics = self._train_one_epoch(epoch)

            self.logger.info_important(
                f" Train: Epoch {epoch} - Time {p.times[-1]:.1f} -"
                f" Reward {metrics['mean_reward']:.3f} -"
                f" N_Actions {metrics['mean_n_actions']:.3f} -"
                f" AS bACC {(metrics['as']['bACC'] if 'as' in metrics else 0):.3f} -"
                f" EF bACC {(metrics['ef_cat']['bACC'] if 'ef_cat' in metrics else 0):.3f} -"
                f" EF MAE {(metrics['ef_mae'] if 'ef_mae' in metrics else 0):.3f}")

            for profiler in self.profilers.values():
                if self.config.log_profile:
                    profiler.log()
                profiler.reset()

            if self.config.use_wandb:
                wandb.log({'epoch': epoch,
                           'train_n_actions_per_epoch': metrics['mean_n_actions'],
                           'train_reward_per_epoch': metrics['mean_reward'],
                           'train_AS_bACC_per_epoch': metrics['as']['bACC'] if 'as' in metrics else 0,
                           'train_EF_bACC_per_epoch': metrics['ef_cat']['bACC'] if 'ef_cat' in metrics else 0,
                           'train_EF_MAE_per_epoch': metrics['ef_mae'] if 'ef_mae' in metrics else 0,
                           'learning_rate': self.agent.scheduler.get_last_lr()[
                               0] if self.config.agent.agent_name != 'TabularQ' else 0.0,
                           'train_epoch_time(s)': p.times[-1]})
                wandb.save(os.path.join(self.save_dir, 'output.log'))

            if self.config.agent.agent_name != 'TabularQ':
                self.agent.scheduler.step()
            ###################################################################

            ######################## validate one epoch #######################
            self.env.data = self.data_val
            with Profiler() as p:
                metrics = self._validate_one_epoch(epoch)

            self.logger.info_important(
                f" Val: Epoch {epoch} - {p.times[-1]:.0f}s -"
                f" Reward {metrics['mean_reward']:.3f} -"
                f" N_Actions {metrics['mean_n_actions']:.3f} -"
                f" AS bACC {(metrics['as']['bACC'] if 'as' in metrics else 0):.3f} -"
                f" EF bACC {(metrics['ef_cat']['bACC'] if 'ef_cat' in metrics else 0):.3f} -"
                f" AS F1 {(metrics['as']['F1'] if 'as' in metrics else 0):.3f} -"
                f" EF F1 {(metrics['ef_cat']['F1'] if 'ef_cat' in metrics else 0):.3f} -"
                f" EF MAE {(metrics['ef_mae'] if 'ef_mae' in metrics else 0):.3f}")

            for profiler in self.profilers.values():
                if self.config.log_profile:
                    profiler.log()
                profiler.reset()

            if self.config.use_wandb:
                wandb.log({
                    'val_n_actions_per_epoch': metrics['mean_n_actions'],
                    'val_reward_per_epoch': metrics['mean_reward'],
                    'val_AS_bACC_per_epoch': metrics['as']['bACC'] if 'as' in metrics else 0,
                    'val_EF_bACC_per_epoch': metrics['ef_cat']['bACC'] if 'ef_cat' in metrics else 0,
                    'val_EF_MAE_per_epoch': metrics['ef_mae'] if 'ef_mae' in metrics else 0,
                    'val_epoch_time(s)': p.times[-1]})
                wandb.save(os.path.join(self.save_dir, 'output.log'))

            # Save model for best reward
            if metrics['mean_reward'] >= max_reward or (epoch + 1) % 5 == 0:
                self.agent.save(self.save_dir, epoch, name=
                f"n_actions_{metrics['mean_n_actions']:.2f}"
                f"_as_bACC_{metrics['as']['bACC']:.3f}"
                f"_ef_bACC_{metrics['ef_cat']['bACC']:.3f}"
                f"_as_F1_{metrics['as']['F1']:.3f}"
                f"_ef_F1_{metrics['ef_cat']['F1']:.3f}"
                f"_ef_MAE_{metrics['ef_mae']:.2f}"
                f"_reward_{metrics['mean_reward']:.2f}"
                f"_epoch_{epoch}")
                max_reward = metrics['mean_reward']

            # if (epoch + 1) % 10 == 0:
            #     self.agent.save(self.save_dir, epoch, mode='latest')
            ###################################################################

            ######################## evaluate on test set #######################
            self.env.data = self.data_test
            with Profiler() as p:
                metrics = self._validate_one_epoch(epoch)

            self.logger.info_important(
                f" Test: Epoch {epoch} - {p.times[-1]:.0f}s -"
                f" Reward {metrics['mean_reward']:.3f} -"
                f" N_Actions {metrics['mean_n_actions']:.3f} -"
                f" AS bACC {(metrics['as']['bACC'] if 'as' in metrics else 0):.3f} -"
                f" EF bACC {(metrics['ef_cat']['bACC'] if 'ef_cat' in metrics else 0):.3f} -"
                f" AS F1 {(metrics['as']['F1'] if 'as' in metrics else 0):.3f} -"
                f" EF F1 {(metrics['ef_cat']['F1'] if 'ef_cat' in metrics else 0):.3f} -"
                f" EF MAE {(metrics['ef_mae'] if 'ef_mae' in metrics else 0):.3f}")

            for profiler in self.profilers.values():
                if self.config.log_profile:
                    profiler.log()
                profiler.reset()
            ###################################################################

    def evaluate(self):
        # set env
        self.config.environment.update({'task': self.config.task, 'device': self.config.device})
        self.env = environment_builder.build(self.config.environment, self.logger, self.classifier, self.data_test)

        # set agent
        self.config.agent.update({
            'state_shape': self.env.state_shape,
            'n_actions': self.env.action_space.n,
            'device': self.config.device})
        self.agent = agent_builder.build(self.config.agent, self.logger)
        self.agent.eps_start, self.agent.eps_final = 0, 0

        if self.config.task in ['AS_EF', 'AS']:
            as_true_all, as_pred_all = [], []
        if self.config.task in ['AS_EF', 'EF']:
            ef_true_all, ef_pred_all, ef_true_cat_all, ef_pred_cat_all = [], [], [], []
        reward_per_episode, n_actions_per_episode = [], []

        for episode_idx in range(self.env.n_episodes()):
            episode = self._run_one_episode(episode_idx, phase='test')
            reward_per_episode.append(episode['episode_reward'])
            n_actions_per_episode.append(len(self.env.action_history))

            if self.config.task in ['AS_EF', 'AS']:
                as_true_all.append(episode['as_label'])
                as_pred_all.append(episode['as_pred'])
            if self.config.task in ['AS_EF', 'EF']:
                ef_true_all.append(episode['ef_value_label'])
                ef_pred_all.append(episode['ef_value_pred'])
                ef_true_cat_all.append(episode['ef_cat_label'])
                ef_pred_cat_all.append(episode['ef_cat_pred'])

            if self.config.task == 'AS_EF':
                self.logger.info(
                    f" Test - Episode {episode_idx} - Reward {episode['episode_reward']:.3f} - Actions {self.env.action_history}"
                    f" - as_label {episode['as_label']} - as_pred {episode['as_pred']} - ef_cat_label {episode['ef_cat_label']} - ef_cat_pred {episode['ef_cat_pred']}"
                    f" - {episode['info']}")
            elif self.config.task == 'AS':
                self.logger.info(
                    f" Test - Episode {episode_idx} - Reward {episode['episode_reward']:.3f} - Actions {self.env.action_history}"
                    f" - as_label {episode['as_label']} - as_pred {episode['as_pred']}"
                    f" - {episode['info']}")
            elif self.config.task == 'EF':
                self.logger.info(
                    f" Test - Episode {episode_idx} - Reward {episode['episode_reward']:.3f} - Actions {self.env.action_history}"
                    f" - ef_cat_label {episode['ef_cat_label']} - ef_cat_pred {episode['ef_cat_pred']}"
                    f" - {episode['info']}")
            else:
                raise ValueError (f"Unknown task {self.config.task}")

            if self.config.use_wandb:
                wandb.log({'test_episode': episode_idx,
                           'test_n_actions_per_episode': n_actions_per_episode[-1],
                           'test_reward_per_episode': reward_per_episode[-1]})

        # Calculate and plot metrics
        self.logger.info_important(f"Mean n_actions {np.mean(n_actions_per_episode):.2f}")
        if self.config.task in ['AS_EF', 'AS']:
            as_metrics = get_classification_metrics(as_true_all, as_pred_all)
            self.logger.info_important(
                f"AS bACC {as_metrics['bACC']:.3f} - "
                f"AS F1 {as_metrics['F1']:.3f} - ")
        if self.config.task in ['AS_EF', 'EF']:
            ef_cat_metrics = get_classification_metrics(ef_true_cat_all, ef_pred_cat_all)
            ef_mae = mean_absolute_error(ef_true_all, ef_pred_all)
            self.logger.info_important(
                f"EF bACC {ef_cat_metrics['bACC']:.3f} - "
                f"EF F1 {ef_cat_metrics['F1']:.3f} - "
                f"EF MAE {ef_mae:.3f}")

        for profiler in self.profilers.values():
            if self.config.log_profile:
                profiler.log()
            profiler.reset()

    def _train_one_epoch(self, epoch: int):
        # shuffle order of episodes in epoch
        episodes_idx = list(range(self.env.n_episodes()))
        random.shuffle(episodes_idx)

        if self.config.task in ['AS_EF', 'AS']:
            as_true_all, as_pred_all = [], []
        if self.config.task in ['AS_EF', 'EF']:
            ef_true_all, ef_pred_all, ef_true_cat_all, ef_pred_cat_all = [], [], [], []
        reward_per_episode, n_actions_per_episode = [], []

        for episode_idx in episodes_idx:
            episode = self._run_one_episode(episode_idx, phase='train')
            reward_per_episode.append(episode['episode_reward'])
            n_actions_per_episode.append(len(self.env.action_history))

            if self.config.task in ['AS_EF', 'AS']:
                as_true_all.append(episode['as_label'])
                as_pred_all.append(episode['as_pred'])
            if self.config.task in ['AS_EF', 'EF']:
                ef_true_all.append(episode['ef_value_label'])
                ef_pred_all.append(episode['ef_value_pred'])
                ef_true_cat_all.append(episode['ef_cat_label'])
                ef_pred_cat_all.append(episode['ef_cat_pred'])

            if (epoch + 1) % 50 == 0:
                epsilon_str = f" - Epsilon {self.agent.eps:.2f}" if issubclass(type(self.agent), DDQNAgent) else ""
                self.logger.info(
                    f"Epoch {epoch} - Episode {episode_idx} - Reward {episode['episode_reward']:.3f} - "
                    f"Actions {self.env.action_history}{epsilon_str} {episode['info']}")

            if self.config.use_wandb:
                wandb.log({'batch_loss': episode['batch_loss'],
                           'episode': episode_idx,
                           'train_n_actions_per_episode': n_actions_per_episode[-1],
                           'train_reward_per_episode': reward_per_episode[-1]})

        metrics = {
            'mean_reward': np.mean(reward_per_episode),
            'mean_n_actions': np.mean(n_actions_per_episode),
        }
        if self.config.task in ['AS_EF', 'AS']:
            metrics['as'] = get_classification_metrics(as_true_all, as_pred_all)
        if self.config.task in ['AS_EF', 'EF']:
            metrics['ef_cat'] = get_classification_metrics(ef_true_cat_all, ef_pred_cat_all)
            metrics['ef_mae'] = mean_absolute_error(ef_true_all, ef_pred_all)

        return metrics

    def _validate_one_epoch(self, epoch: int):
        if self.config.task in ['AS_EF', 'AS']:
            as_true_all, as_pred_all = [], []
        if self.config.task in ['AS_EF', 'EF']:
            ef_true_all, ef_pred_all, ef_true_cat_all, ef_pred_cat_all = [], [], [], []
        reward_per_episode, n_actions_per_episode = [], []

        for episode_idx in range(self.env.n_episodes()):
            episode = self._run_one_episode(episode_idx, phase='val')
            reward_per_episode.append(episode['episode_reward'])
            n_actions_per_episode.append(len(self.env.action_history))

            if self.config.task in ['AS_EF', 'AS']:
                as_true_all.append(episode['as_label'])
                as_pred_all.append(episode['as_pred'])
            if self.config.task in ['AS_EF', 'EF']:
                ef_true_all.append(episode['ef_value_label'])
                ef_pred_all.append(episode['ef_value_pred'])
                ef_true_cat_all.append(episode['ef_cat_label'])
                ef_pred_cat_all.append(episode['ef_cat_pred'])

            if (epoch + 1) % 50 == 0:
                epsilon_str = f" - Epsilon {self.agent.eps:.2f}" if issubclass(type(self.agent), DDQNAgent) else ""
                self.logger.info(
                    f"Epoch {epoch} - Episode {episode_idx} - Reward {episode['episode_reward']:.3f} - "
                    f"Actions {self.env.action_history}{epsilon_str} {episode['info']}")

        metrics = {
            'mean_reward': np.mean(reward_per_episode),
            'mean_n_actions': np.mean(n_actions_per_episode),
        }
        if self.config.task in ['AS_EF', 'AS']:
            metrics['as'] = get_classification_metrics(as_true_all, as_pred_all)
        if self.config.task in ['AS_EF', 'EF']:
            metrics['ef_cat'] = get_classification_metrics(ef_true_cat_all, ef_pred_cat_all)
            metrics['ef_mae'] = mean_absolute_error(ef_true_all, ef_pred_all)

        return metrics

    def _run_one_episode(self, episode_idx: int, phase: str) -> dict:
        state, *_ = self.env.reset(episode_idx)
        done = False
        episode_reward = 0
        agent_loss = []

        while not done:
            # take action
            with self.profilers['Agent Taking Actions']:
                if isinstance(self.agent, DDQNAgent):
                    action = self.agent.take_one_action(state, self.n_transitions)
                elif isinstance(self.agent, (PPOAgent, GRPOAgent)):
                    action = self.agent.take_one_action(state, phase)
                elif self.config.agent.agent_name == 'TabularQ':
                    action = self.agent.take_one_action(state, self.n_transitions)

            # take one step
            with self.profilers['Classifier Predictions'] as profiler:
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                if not done:
                    profiler.skip()

            if phase == 'train':
                # store transition
                with self.profilers['Agent Store Transitions']:
                    self.agent.store_transition(state, action, reward, next_state, done)
                self.n_transitions += 1

                # update agent
                if isinstance(self.agent, DDQNAgent):
                    if self.n_transitions >= self.agent.memory_bs:
                        with self.profilers['Agent Updates']:
                            agent_loss.append(self.agent.update_policy_net())
                    if self.n_transitions % self.config.agent.target_update_freq == 0:
                        self.agent.update_target_net()
                elif isinstance(self.agent, (PPOAgent, GRPOAgent)):
                    if self.n_transitions % self.config.agent.update_freq == 0:
                        with self.profilers['Agent Updates']:
                            agent_loss.append(self.agent.update())
                elif self.config.agent.agent_name == 'TabularQ':
                    self.agent.update_table(state, action, reward, next_state)

            episode_reward += reward
            state = next_state

        mean_loss = np.mean(agent_loss) if agent_loss else 0

        out = {'episode_reward': episode_reward,
               'mean_loss': mean_loss,
               'info': info, }
        if self.config.task in ['AS_EF', 'AS']:
            out['as_label'] = self.env.as_label
            out['as_pred'] = self.env.as_pred
        if self.config.task in ['AS_EF', 'EF']:
            out['ef_cat_label'] = self.env.ef_cat_label
            out['ef_cat_pred'] = self.env.ef_cat_pred
            out['ef_value_label'] = self.env.ef_value_label
            out['ef_value_pred'] = self.env.ef_value_pred
        out['joint_pred_pmf'] = self.env.joint_pred_pmf

        return out
