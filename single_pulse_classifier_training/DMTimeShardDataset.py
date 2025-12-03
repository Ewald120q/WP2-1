import json
import os
import torch
import numpy as np


def _resolve_manifest(directory, preferred_names=None):
    if preferred_names is None:
        preferred_names = []
    for name in preferred_names:
        if not name:
            continue
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate):
            return candidate
    # Default "manifest.json"
    manifest_json = os.path.join(directory, "manifest.json")
    if os.path.isfile(manifest_json):
        return manifest_json
    # Fallback: any json file containing "manifest"
    if os.path.isdir(directory):
        for entry in os.listdir(directory):
            if entry.endswith('.json') and 'manifest' in entry:
                candidate = os.path.join(directory, entry)
                if os.path.isfile(candidate):
                    return candidate
    raise FileNotFoundError(f"No manifest file found in {directory}")

class ShardIndex:
    def __init__(self, manifest_path, root_dir):
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.root_dir = root_dir
        self.entries = manifest["shards"]
        self.total = manifest["total_samples"]

    def locate(self, global_idx):
        for entry in self.entries:
            if entry["start_sample"] <= global_idx < entry["end_sample"]:
                local = global_idx - entry["start_sample"]
                path = os.path.join(self.root_dir, entry["path"])
                return path, local
        raise IndexError(global_idx)
    
class DMTimeShardDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, use_freq_time=False, dtype=torch.float32, split="train"):
        if split not in {"train", "test"}:
            raise ValueError(f"Unsupported split '{split}'. Expected 'train' or 'test'.")

        dm_split_dir = os.path.join(cfg["output_dir"], "dm_time_shards", split)
        freq_split_dir = os.path.join(cfg["output_dir"], "dedispersed_freq_time_shards", split)

        dm_manifest = _resolve_manifest(
            dm_split_dir,
            preferred_names=[f"{cfg['prefix']}_DM_time_dataset_manifest.json"]
        )

        freq_manifest = None
        if use_freq_time:
            freq_manifest = _resolve_manifest(
                freq_split_dir,
                preferred_names=[f"{cfg['prefix']}_dedispersed_freq_time_manifest.json"]
            )

        self.dm_index = ShardIndex(dm_manifest, dm_split_dir)
        self.use_freq_time = use_freq_time
        self.freq_index = (ShardIndex(freq_manifest, freq_split_dir)
                           if use_freq_time else None)

        labels_filename = f"{cfg['prefix']}_DM_time_dataset_realbased_labels_{split}.npy"
        metadata_filename = f"{cfg['prefix']}_DM_time_dataset_realbased_metadata_{split}.npy"

        self.labels = np.load(os.path.join(cfg["output_dir"], labels_filename), mmap_mode='r')
        self.metadata = np.load(os.path.join(cfg["output_dir"], metadata_filename), mmap_mode='r')
        self.dtype = dtype
        self._dm_cache = {}
        self._freq_cache = {}

    def __len__(self):
        return self.dm_index.total

    def _read_sample(self, cache, shard_index, global_idx):
        path, local_idx = shard_index.locate(global_idx)
        if path not in cache:
            cache[path] = np.load(path, mmap_mode='r')
        return cache[path][local_idx]

    def __getitem__(self, idx):
        dm_patch = self._read_sample(self._dm_cache, self.dm_index, idx).copy()
        dm_tensor = torch.from_numpy(dm_patch).to(self.dtype)

        result = {"dm_time": dm_tensor,
                  "label": torch.tensor(self.labels[idx], dtype=torch.long),
                  "metadata": torch.from_numpy(self.metadata[idx].copy()).float()}

        if self.use_freq_time:
            freq_patch = self._read_sample(self._freq_cache, self.freq_index, idx)
            result["freq_time"] = torch.from_numpy(freq_patch.copy()).to(self.dtype)

        return result