from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import training
from training_models import models_htable

try:
    from .finetune_dataset import load_index_dataset
except ImportError:
    from finetune_dataset import load_index_dataset


MODEL_INPUT_CONFIG = {
    "f_small": {"mode": "dmt", "use_frequency": False},
    "f_mid": {"mode": "ft", "use_frequency": True},
    "f_large": {"mode": "dmft", "use_frequency": True},
}


def _make_loader(dataset, config, *, shuffle):
    num_workers = config.get("num_workers", 0)
    kwargs = {
        "batch_size": config["batch_size"],
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = config.get("prefetch_factor", 2)
    return DataLoader(dataset, **kwargs)


def train_finetune(config):
    """Train like training.train(), using route datasets and a baseline checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    route_name = config["route_name"]
    model_name = config["model_name"]
    try:
        input_config = MODEL_INPUT_CONFIG[route_name]
    except KeyError as exc:
        supported_models = ", ".join(MODEL_INPUT_CONFIG)
        raise ValueError(
            f"Unknown route_name {route_name!r}. Expected one of: {supported_models}"
        ) from exc

    mode = input_config["mode"]
    use_frequency = input_config["use_frequency"]
    datasets_dir = Path(config["path_to_finetune_datasets"])

    train_dataset = load_index_dataset(
        datasets_dir / f"{route_name}_train_finetune.npz",
        data_root=config["path_to_files"],
        dataset_prefix=config["dataset_prefix"],
        use_freq_time=use_frequency,
    )
    val_dataset = load_index_dataset(
        datasets_dir / f"{route_name}_val_route.npz",
        data_root=config["path_to_files"],
        dataset_prefix=config["dataset_prefix"],
        use_freq_time=use_frequency,
    )
    train_loader = _make_loader(train_dataset, config, shuffle=True)
    val_loader = _make_loader(val_dataset, config, shuffle=False)

    model = models_htable[model_name](
        config["resolution"], mode, config["dropout"], device
    ).to(device)
    checkpoint = torch.load(config["pretrained_checkpoint"], map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=5)

    tb_cfg = config["tensorboard"]
    log_dir = (
        Path(tb_cfg["log_root"])
        / tb_cfg["experiment_name"]
        / tb_cfg["run_name"]
    )
    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_text("run/config", json.dumps(config, indent=2), 0)

    checkpoint_dir = (
        Path(config["path_to_checkpoints"])
        / route_name
        / tb_cfg["experiment_name"]
        / tb_cfg["run_name"]
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print(f"Device: {device}")
    print(f"Loaded baseline: {config['pretrained_checkpoint']}")
    print(f"Train dataset: {len(train_dataset):,}")
    print(f"Validation route: {len(val_dataset):,}")

    try:
        return training._train(
            model,
            f"{model_name}_finetune",
            train_loader,
            val_loader,
            optimizer,
            config["num_epochs"],
            criterion=nn.CrossEntropyLoss(),
            scheduler=scheduler,
            writer=writer,
            device=device,
            patience=config["patience"],
            checkpoint_dir=str(checkpoint_dir),
        )
    finally:
        writer.close()
