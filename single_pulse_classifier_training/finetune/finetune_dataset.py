from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from DMTimeShardDataset import DMTimeShardDataset
from training_utils import label_encoding


class IndexedFineTuneDataset(Dataset):
    """View of a DMTimeShardDataset using the generated fine-tuning indices."""

    def __init__(self, base_dataset, indices, is_replay=None):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        if is_replay is None:
            is_replay = np.zeros(len(self.indices), dtype=np.uint8)
        self.is_replay = np.asarray(is_replay, dtype=np.uint8)

        if len(self.indices) != len(self.is_replay):
            raise ValueError("indices and is_replay must have equal length")
        if len(self.indices) and (
            self.indices.min() < 0 or self.indices.max() >= len(base_dataset)
        ):
            raise IndexError("Fine-tuning indices are outside the source dataset")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = int(self.indices[index])
        sample = dict(self.base_dataset[source_index])
        sample["source_index"] = torch.tensor(source_index, dtype=torch.long)
        sample["is_replay"] = torch.tensor(
            bool(self.is_replay[index]), dtype=torch.bool
        )
        return sample


def load_index_dataset(
    dataset_file,
    *,
    data_root,
    dataset_prefix,
    use_freq_time=True,
):
    """Load a generated `.npz` dataset against the existing shard dataset."""
    dataset_file = Path(dataset_file)
    saved = np.load(dataset_file, allow_pickle=False)
    split = str(saved["source_split"].item())

    base_dataset = DMTimeShardDataset(
        {"output_dir": str(data_root), "prefix": dataset_prefix},
        use_freq_time=use_freq_time,
        split=split,
    )
    base_dataset.labels = label_encoding(base_dataset.labels.astype(object))

    expected_size = int(saved["source_dataset_size"])
    if len(base_dataset) != expected_size:
        raise ValueError(
            f"{dataset_file.name} expects {expected_size} source samples, "
            f"but {data_root!s}/{split} contains {len(base_dataset)}"
        )

    return IndexedFineTuneDataset(
        base_dataset,
        indices=saved["indices"],
        is_replay=saved["is_replay"],
    )
