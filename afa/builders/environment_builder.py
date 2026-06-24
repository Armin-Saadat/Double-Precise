from afa.environments import EchoPrimeEnv


def build(config, logger, classifier, data):
    env_builders = {    
        'Echo-Prime': _build_echo_prime,        
    }
    if config.env_name not in env_builders.keys():
        logger.error('No environment named {}.'.format(config.env_name))
    env = env_builders[config.env_name](config, logger, classifier, data)
    logger.warning('Environment {} is created successfully.'.format(config.env_name))

    return env


def _build_echo_prime(config, logger, classifier, data):
    return EchoPrimeEnv(data=data,
                        n_actions=config.n_actions,
                        classifiers=classifier,
                        lammy=config.lammy,
                        max_step=config.max_step,
                        reward_scheme=config.reward_scheme,
                        task=config.task,
                        device=config.device,
                        logger=logger)
