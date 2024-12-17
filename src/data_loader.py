import os
from pathlib import Path

import numpy as np

import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


class LightningMoleDataModule(L.LightningDataModule):
    """
    A PyTorch Lightning DataModule.

    Args:
        path: Path to the npy file with preprocessed moles.
        batch_size: Batch size for DataLoaders.
    """

    def __init__(self, path: str | Path | np.ndarray, batch_size: int = 2048, seed: int = 42):
        super().__init__()
        self.data_path = path
        self.batch_size = batch_size
        self.seed = seed

        if isinstance(path, np.ndarray):
            data = path
        else:
            data = np.load(self.data_path)

        self.seq_length = data.shape[1]
        if data.ndim == 2:  # token indices
            #TODO: fix it (what if there is no unknown character)
            self.vocab_size = len(np.unique(data))
        elif data.ndim == 3:  # one-hot encoded
            self.vocab_size = data.shape[2]

        self.data = data
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def setup(self, stage: str = None):
        """
        Set up the dataset splits for training, validation, and testing.
        """

        dataset = TensorDataset(torch.from_numpy(self.data).float())
        total_size = len(dataset)
        train_size = int(total_size * 0.8)
        val_size = int(total_size * 0.1)
        test_size = total_size - train_size - val_size  # Ensure no data is lost due to rounding
        self.train_data, self.val_data, self.test_data = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.seed)
        )

    def train_dataloader(self) -> DataLoader:
        """
        Returns the training DataLoader.
        """
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,
                          num_workers=min(2, os.cpu_count() // 2),
                          persistent_workers=True
                          )

    def val_dataloader(self) -> DataLoader:
        """
        Returns the validation DataLoader.
        """
        return DataLoader(self.val_data, batch_size=self.batch_size,
                          num_workers=min(2, os.cpu_count() // 2),
                          persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        """
        Returns the test DataLoader.
        """
        return DataLoader(self.test_data, batch_size=self.batch_size,
                          num_workers=min(2, os.cpu_count() // 2),
                          persistent_workers=True)


def create_data_module(batch_size, path, seed=42):
    dm = LightningMoleDataModule(path=path, batch_size=batch_size, seed=seed)
    return dm, dm.seq_length, dm.vocab_size
