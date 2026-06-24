import os.path
import torch


class EchoPrimeDataset(Dataset):
    def __init__(self, dataset_root: str = '', task: str = 'AS_EF', split: str = "val"):
        self.dataset = torch.load(os.path.join(dataset_root, f""), weights_only=False)
        self.split = split
        self.task = task

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
