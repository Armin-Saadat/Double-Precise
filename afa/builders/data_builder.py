import os
from sklearn.model_selection import train_test_split

from afa.datasets import EchoPrimeDataset


def build(config):
    data_builders = {        
        'Echo-Prime': _build_echo_prime,        
    }

    return data_builders[config.dataset_name](config)


def _build_echo_prime(config):
    data_train = EchoPrimeDataset(dataset_root=config.dataset_path, task=config.task, split='train')
    data_val = EchoPrimeDataset(dataset_root=config.dataset_path, task=config.task, split='val')
    data_test = EchoPrimeDataset(dataset_root=config.dataset_path, task=config.task, split='test')

    return data_train, data_val, data_test
