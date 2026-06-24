import torch
from afa.classifiers import ProtoASSimFormer, ProtoASEFSimFormer, ConfidentMultiFormer


def build(config, logger):
    classifier_builders = {                        
        'echo_prime_former': _build_echo_prime_former,        
    }
    classifier = classifier_builders[config.classifier_name](config, logger)

    return classifier


def _build_echo_prime_former(config, logger):
    classifiers = {}
    if config.task in ['AS_EF', 'AS']:
        if config.as_head == 'sim_former':
            as_classifier = ProtoASSimFormer(n_videos=5, n_sims=512, n_classes=3, logger=logger)
        elif config.as_head == 'multi_former':
            as_classifier = ProtoASEFSimFormer(n_videos=5, n_sims=512, as_n_classes=3, ef_n_classes=1, logger=logger)
        elif config.as_head == 'conf_multi_former':
            as_classifier = ConfidentMultiFormer(n_videos=5, n_sims=512, prob_func=config.prob_func, dist=config.dist,
                                                 sigma_floor=config.sigma_floor)
        else:
            raise ValueError(f'Unknown AS head {config.as_head}')
        assert config.as_classifier_path is not None, f'You have to pass the checkpoint path for AS classifier'
        as_classifier.load_state_dict(
            torch.load(config.as_classifier_path, map_location=config.device)['model_state_dict'])
        as_classifier.to(config.device)
        as_classifier.eval()
        classifiers['AS'] = as_classifier
        classifiers['as_head'] = config.as_head
    if config.task in ['AS_EF', 'EF']:
        if config.ef_head == 'sim_former':
            ef_regressor = ProtoASSimFormer(n_videos=5, n_sims=512, n_classes=1, logger=logger)
        elif config.ef_head == 'multi_former':
            ef_regressor = ProtoASEFSimFormer(n_videos=5, n_sims=512, as_n_classes=3, ef_n_classes=1, logger=logger)
        elif config.ef_head == 'conf_multi_former':
            ef_regressor = ConfidentMultiFormer(n_videos=5, n_sims=512, prob_func=config.prob_func, dist=config.dist,
                                                sigma_floor=config.sigma_floor)
        else:
            raise ValueError(f'Unknown EF head {config.ef_head}')
        assert config.ef_regressor_path is not None, f'You have to pass the checkpoint path for EF regressor'
        ef_regressor.load_state_dict(
            torch.load(config.ef_regressor_path, map_location=config.device)['model_state_dict'])
        ef_regressor.to(config.device)
        ef_regressor.eval()
        classifiers['EF'] = ef_regressor
        classifiers['ef_head'] = config.ef_head

    return classifiers
