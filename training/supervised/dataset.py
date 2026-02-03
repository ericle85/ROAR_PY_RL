"""Dataset class for loading expert demonstration data."""

import glob
import os
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class ExpertDataset(Dataset):
    """Dataset for expert demonstration data stored in .npz files.

    Args:
        data_dirs: Directory or list of directories containing .npz files
            with 'observations' and 'actions' arrays.
    """

    def __init__(self, data_dirs: Union[str, List[str]] = "training/expert_data"):
        # Normalize to list
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        self.data_dirs = data_dirs

        # Load all .npz files from all directories
        observations_list = []
        actions_list = []
        total_files = 0

        for data_dir in data_dirs:
            npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
            if npz_files:
                print(f"Loading from {data_dir}: {len(npz_files)} files")
                for npz_file in npz_files:
                    data = np.load(npz_file)
                    observations_list.append(data["observations"])
                    actions_list.append(data["actions"])
                total_files += len(npz_files)

        if total_files == 0:
            raise ValueError(f"No .npz files found in {data_dirs}")

        # Concatenate all data
        self.observations = torch.from_numpy(
            np.concatenate(observations_list, axis=0)
        ).float()
        self.actions = torch.from_numpy(
            np.concatenate(actions_list, axis=0)
        ).float()

        assert len(self.observations) == len(self.actions), (
            f"Observation count ({len(self.observations)}) != action count ({len(self.actions)})"
        )

        print(f"Loaded {len(self)} transitions from {total_files} files")
        print(f"Observation shape: {self.observations.shape}")
        print(f"Action shape: {self.actions.shape}")

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.observations[idx], self.actions[idx]


def create_dataloaders(
    data_dir: str = "training/expert_data",
    batch_size: int = 256,
    val_split: float = 0.2,
    seed: int = 1,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders from expert data.

    Args:
        data_dir: Directory containing .npz files.
        batch_size: Batch size for DataLoaders.
        val_split: Fraction of data to use for validation.
        seed: Random seed for reproducible splits.
        num_workers: Number of worker processes for data loading.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    dataset = ExpertDataset(data_dir)

    # Calculate split sizes
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    # Create reproducible split
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Train size: {train_size}, Val size: {val_size}")

    return train_loader, val_loader
