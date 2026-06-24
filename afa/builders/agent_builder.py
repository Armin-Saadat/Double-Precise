from afa.agents import EchoPrimePPOAgent


def build(config, logger):
    agent_builders = {        
        'Echo-Prime-PPO': _build_echo_prime_ppo,        
    }
    if config.agent_name not in agent_builders.keys():
        logger.error('No agent named {}.'.format(config.agent_name))
    agent = agent_builders[config.agent_name](config, logger)
    logger.warning('Agent {} is created successfully.'.format(config.agent_name))
    if config.agent_checkpoint:
        agent.load(config.agent_checkpoint)
        logger.warning(f'Agent is loaded successfully from checkpoint at {agent.checkpoint_epoch} epoch.')

    return agent


def _build_echo_prime_ppo(config, logger):
    return EchoPrimePPOAgent(state_shape=config.state_shape,
                             n_actions=config.n_actions,
                             lr=config.lr,
                             milestones=config.milestones,
                             gamma=config.gamma,
                             discount_factor=config.discount_factor,
                             update_n_epochs=config.update_n_epochs,
                             device=config.device,
                             logger=logger)
