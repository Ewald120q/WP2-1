import os
import torch
import numpy as np
import random
import argparse
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter

import training_models
import training
from rejector import Rejector, EmbeddingRejector
from ensemble import TorchRejectionEnsemble
from DMTimeShardDataset import DMTimeShardDataset
from skrejector import SNRDT_Rejector
from embedding_processing_models import build_embedding_processing


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def get_rejector_state_dict(ckpt):
    """Normalize different checkpoint formats to the rejector model's state_dict."""
    state_dict = extract_state_dict(ckpt)
    if not isinstance(state_dict, dict):
        return state_dict

    clean_dict = {}
    for k, v in state_dict.items():
        if k.startswith("1.net."):
            clean_dict[k.replace("1.net.", "net.")] = v
        elif k.startswith("1."):
            clean_dict[k.replace("1.", "")] = v
        elif k.startswith("embedding_processing."):
            clean_dict[k.replace("embedding_processing.", "")] = v

    return clean_dict or state_dict

def print_runtime_diagnostics(config):
    print("--- Runtime diagnostics ---")
    print(f"Configured device: {config['device']}")
    print(f"torch: {torch.__version__}")
    print(f"torch cuda available: {torch.cuda.is_available()}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(
        "DataLoader: "
        f"batch_size={config['batch_size']}, "
        f"num_workers={config.get('num_workers')}, "
        f"pin_memory={config.get('pin_memory')}, "
        f"persistent_workers={config.get('persistent_workers')}, "
        f"prefetch_factor={config.get('prefetch_factor')}"
    )
    if torch.cuda.is_available():
        current = torch.cuda.current_device()
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {current} ({torch.cuda.get_device_name(current)})")
        free_bytes, total_bytes = torch.cuda.mem_get_info(current)
        print(
            "CUDA memory free/total: "
            f"{free_bytes / 1024**3:.2f} GiB / {total_bytes / 1024**3:.2f} GiB"
        )
    else:
        print("WARNING: CUDA is not available in this process. Training will run on CPU.")
    print("---------------------------")

parser = argparse.ArgumentParser(description="Rejector Random Search Worker")
parser.add_argument("--worker_id", type=int, help="Eindeutige ID für diesen Container")
parser.add_argument("--rejector_type", type=str, choices=["standard", "embedding", "snrdt", "random"], help="Art des Rejectors, der trainiert werden soll.")
args = parser.parse_args()
#/raid/outputs
#/cephfs/users/oleksjuk/MA/WP2-1/DM_time_dataset_creator/outputs
_config = {
    "dataset_cfg": {
        "output_dir": "/raid/outputs",
        "prefix": "B0531+21_59000_48386",
    },
    "batch_size": 1024,
    "num_workers": 10,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 2,
    
    #no finetune
    
    # "f_small_weights": "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/final_checkpoints/baseline_rejection_ensemble/prot-DM_time_binary_classificator_241002_3_GAP-014-0.764-0.740.pth",
    # "f_mid_weights": "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/final_checkpoints/baseline_rejection_ensemble/prot-DM_time_binary_classificator_241002_5_GAP-060-0.973-0.948.pth",
    #"f_big_weights": "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/final_checkpoints/baseline_rejection_ensemble/prot-DM_time_binary_classificator_resnet18-003-0.993-0.993.pth",
    
    
    #finetuned models
    "f_small_weights": "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/final_checkpoints/finetune_checkpoints/prot-DM_time_binary_classificator_241002_3_GAP_finetune-004-0.838-0.813.pth",
    "f_mid_weights": "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/final_checkpoints/finetune_checkpoints/prot-DM_time_binary_classificator_241002_5_GAP_finetune-019-0.989-0.993.pth",
    "f_big_weights": "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/final_checkpoints/finetune_checkpoints/prot-DM_time_binary_classificator_resnet18_finetune-010-0.999-0.993.pth",
    
    #no finetune
    # "splits_path": "./new_balanced_dm_time_splits_cascaded_r1.pth",
    #"splits_path": "./new_balanced_dm_time_splits_cascaded_r2_r1filtered.pth",
    #"unbalanced_splits_path": "./new_unbalanced_dm_time_splits_cascaded_r2_r1filtered.pth",
    
    #finetune
    # "splits_path": "./new_balanced_dm_time_splits_cascaded_r1_FINETUNE.pth",
    # "unbalanced_splits_path": "./new_unbalanced_dm_time_splits_cascaded_r1_r1filtered_FINETUNE.pth",
    "splits_path": "./new_balanced_dm_time_splits_cascaded_r2_r1filtered_FINETUNE.pth",
    "unbalanced_splits_path": "./new_unbalanced_dm_time_splits_cascaded_r2_r1filtered_FINETUNE.pth",
    
    # Which stage targets/splits to create when splits_path doesn't exist.
    # - "r1": old behavior (small->mid) using prepare_fit + _splitTrainData
    # - "r2": new behavior (mid->large) using prepare_fit_r2 (r1-filtered subset) + _splitTrainData
    "routing_stage": "r2",
    "tensorboard_root": "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/tensorboard_runs_rejector_r2_FINETUNE/",
    "num_trials": 1,
    "num_epochs": 100,
    "patience": 15,
    "learning_rate": 4.51e-05,
    "weight_decay": 0.0,
    "embedding_model_choices": ["conv_mlp"],# "conv_gap"], "pooled_mlp_128_64", "pooled_mlp_256_256"
    "cnn_channels_choices": [64],
    "extra_conv_choices": [True],
    "pool_size_choices": [7],
    "pool_type_choices": ["max", "avg"],
    "hidden_dim_choices": [128],
    "dropout_choices": [0.2],
    "use_meta_snr_choices": [False], #just for SNRDT
    "use_freq_time": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}





def load_data(config):
    device = config["device"]
    routing_stage = str(config.get("routing_stage", "r2")).lower()
    if routing_stage not in {"r1", "r2"}:
        raise ValueError(f"Unknown routing_stage={routing_stage!r}. Use 'r1' or 'r2'.")

    print("Loading base models to prepare routing splits...")
    use_freq_time = config.get("use_freq_time", False)
    
    # Load small and big models
    small_model = training_models.models_htable["DM_time_binary_classificator_241002_3_GAP"](256, mode="dmt", dropout=False, device=device).to(device)
    small_model.load_state_dict(torch.load(config["f_small_weights"], map_location=device)["model_state_dict"])
    small_model.eval()
    
    mid_model = training_models.models_htable['DM_time_binary_classificator_241002_5_GAP'](256, mode="ft", dropout=False, device=device).to(device)
    mid_model.load_state_dict(torch.load(config["f_mid_weights"], map_location=device)["model_state_dict"])
    mid_model.eval()

    big_model = training_models.models_htable['DM_time_binary_classificator_resnet18'](256, mode="dmft", dropout=False, device=device).to(device)
    big_model.load_state_dict(torch.load(config["f_big_weights"], map_location=device)["model_state_dict"])
    big_model.eval()

    # A fixed, already trained R1 is only needed to select the R2 input subset.
    r1 = None
    if routing_stage == "r2":
        # r1_ckpt_path = "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/final_checkpoints/baseline_rejection_ensemble/prot-run_embedding_3_GAP_conv_mlp_lr1.05e-05_wd0.00e+00_drop0.0_channels64_extraFalse_pool7_hidden64_worker2_trial3-042-0.712-0.635.pth"
        r1_ckpt_path = "/cephfs/users/oleksjuk/MA/WP2-1/single_pulse_classifier_training/final_checkpoints/finetune_checkpoints/prot-run_embedding_r1_conv_mlp_lr1.05e-05_wd0.00e+00_drop0.0_channels64_extraFalse_pool7_hidden64_worker3_trial0-030-0.688-0.618.pth"
        
        r1_kwargs = {
            "model_name": "conv_mlp",
            "cnn_channels": 64,
            "extra_conv": False,
            "pool_size": 7,
            "hidden_dim": 64,
        }

        print("Loading fixed r1 rejector checkpoint for r2 subset selection...")
        embedding_processing, feature_source = build_embedding_processing(
            in_channels=12, **r1_kwargs
        )
        embedding_processing = embedding_processing.to(device)
        embedding_processing.eval()
        r1_ckpt = torch.load(r1_ckpt_path, map_location=device)
        embedding_processing.load_state_dict(get_rejector_state_dict(r1_ckpt))
        r1 = EmbeddingRejector(
            small_model,
            embedding_processing,
            device,
            feature_source=feature_source,
        )
    else:
        print("routing_stage='r1': no existing r1 checkpoint is loaded.")

    rejection_ensemble = TorchRejectionEnsemble(
        small_model,
        mid_model,
        p=0.8,
        rejector=r1,
        calibration=False,
        # reject_threshold_r1=0.548808,
        reject_threshold_r1=0.542446,
    )
    
    print("Loading uncut datasets for evaluation...")
    dataset_cfg = config["dataset_cfg"]
    
    pulse_train_dataset = DMTimeShardDataset(dataset_cfg, use_freq_time=use_freq_time, split="train")
    pulse_train_dataset.labels = training.label_encoding(pulse_train_dataset.labels.astype(object))
    
    pulse_val_dataset = DMTimeShardDataset(dataset_cfg, use_freq_time=use_freq_time, split="val")
    pulse_val_dataset.labels = training.label_encoding(pulse_val_dataset.labels.astype(object))

    pulse_train_loader = DataLoader(
        pulse_train_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 2),
        pin_memory=config.get("pin_memory", True),
        persistent_workers=config.get("persistent_workers", True) and config.get("num_workers", 0) > 0,
        prefetch_factor=config.get("prefetch_factor", 2) if config.get("num_workers", 0) > 0 else None,
    )
    pulse_val_loader = DataLoader(
        pulse_val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 2),
        pin_memory=config.get("pin_memory", True),
        persistent_workers=config.get("persistent_workers", True) and config.get("num_workers", 0) > 0,
        prefetch_factor=config.get("prefetch_factor", 2) if config.get("num_workers", 0) > 0 else None,
    )

    # If splits are already cached, load them directly
    if os.path.exists(config["splits_path"]):# and os.path.exists(config["unbalanced_splits_path"]):
        print(f"Loading cached balanced splits from {config['splits_path']}...")
        #wenn splits mit anderer config erstellt wurde als am ende trainiert wird, übernehmen wir damit die neuen settings
        loader_overrides = {
            "batch_size": config["batch_size"],
            "num_workers": config.get("num_workers", 2),
            "pin_memory": config.get("pin_memory", True),
            "persistent_workers": config.get("persistent_workers", True),
            "prefetch_factor": config.get("prefetch_factor", 2),
        }
        (
            balanced_pulse_train_loader,
            balanced_routing_train_targets,
            balanced_pulse_val_loader,
            balanced_routing_val_targets,
        ) = rejection_ensemble.load_balanced_splits(
            config["splits_path"],
            loader_overrides=loader_overrides,
        )
    else:
        print("Creating balanced routing splits (this takes time)...")
        if routing_stage == "r1":
            # Old behavior: create targets for r1 (small->mid)
            routing_targets_train, routing_targets_val, pulse_train_preds_small, pulse_val_preds_small = rejection_ensemble.prepare_fit(
                pulse_train_loader, pulse_val_loader
            )

            balanced_pulse_train_loader, balanced_routing_train_targets = rejection_ensemble._splitTrainData(
                pulse_train_loader, routing_targets_train, pulse_train_preds_small
            )
            balanced_pulse_val_loader, balanced_routing_val_targets = rejection_ensemble._splitTrainData(
                pulse_val_loader, routing_targets_val, pulse_val_preds_small
            )
        else:
            # New behavior: create targets for r2 (mid->large) on the r1-filtered subset
            (
                routing_targets_train,
                routing_targets_val,
                mid_train_logits_sub,
                mid_val_logits_sub,
                _large_train_logits_sub,
                _large_val_logits_sub,
                _Y_train_sub,
                _Y_val_sub,
                train_mask_r1,
                val_mask_r1,
            ) = rejection_ensemble.prepare_fit_r2(
                pulse_train_loader,
                pulse_val_loader,
                big_model,
                return_debug=True,
            )

            # Build subset loaders that match the lengths of routing_targets_* and *_logits_sub
            train_indices_r2 = np.where(train_mask_r1)[0]
            val_indices_r2 = np.where(val_mask_r1)[0]

            r2_train_loader = DataLoader(
                Subset(pulse_train_dataset, train_indices_r2),
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=config.get("num_workers", 2),
                pin_memory=config.get("pin_memory", True),
                persistent_workers=config.get("persistent_workers", True) and config.get("num_workers", 0) > 0,
                prefetch_factor=config.get("prefetch_factor", 2) if config.get("num_workers", 0) > 0 else None,
            )
            r2_val_loader = DataLoader(
                Subset(pulse_val_dataset, val_indices_r2),
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=config.get("num_workers", 2),
                pin_memory=config.get("pin_memory", True),
                persistent_workers=config.get("persistent_workers", True) and config.get("num_workers", 0) > 0,
                prefetch_factor=config.get("prefetch_factor", 2) if config.get("num_workers", 0) > 0 else None,
            )
            
            rejection_ensemble.save_balanced_splits(
            config["unbalanced_splits_path"],
            r2_train_loader,
            routing_targets_train,
            r2_val_loader,
            routing_targets_val,
            )

            # Balance (use mid logits as "pulse_preds")
            balanced_pulse_train_loader, balanced_routing_train_targets = rejection_ensemble._splitTrainData(
                r2_train_loader, routing_targets_train, mid_train_logits_sub
            )
            balanced_pulse_val_loader, balanced_routing_val_targets = rejection_ensemble._splitTrainData(
                r2_val_loader, routing_targets_val, mid_val_logits_sub
            )
        
        rejection_ensemble.save_balanced_splits(
            config["splits_path"],
            balanced_pulse_train_loader,
            balanced_routing_train_targets,
            balanced_pulse_val_loader,
            balanced_routing_val_targets,
        )
        print(f"Saved balanced loaders to {config['splits_path']}")

    return (
        rejection_ensemble,
        balanced_pulse_train_loader,
        balanced_routing_train_targets,
        balanced_pulse_val_loader,
        balanced_routing_val_targets,
        pulse_train_loader,
        pulse_val_loader,
    )


def get_rejector(rejector_type, config, base_rejection_ensemble, trial_config):
    device = config["device"]
    
    if rejector_type == "standard":
        print("Initializing Standard Rejector (Type 1)...")
        r_model = training_models.models_htable["DM_time_binary_classificator_241002_3_GAP"](256, mode="dmt", dropout=True, device=device).to(device)
        rejector = Rejector(r_model, device)
        
    elif rejector_type == "embedding":
        print("Initializing Embedding Rejector (Type 2)...")
        routing_stage = config.get("routing_stage", "r1")
        if routing_stage == "r2":
            f_base = training_models.models_htable["DM_time_binary_classificator_241002_5_GAP"](256, mode="ft", dropout=False, device=device).to(device)
            f_base.load_state_dict(base_rejection_ensemble.fbig.state_dict())
        else:
            f_base = training_models.models_htable["DM_time_binary_classificator_241002_3_GAP"](256, mode="dmt", dropout=False, device=device).to(device)
            f_base.load_state_dict(base_rejection_ensemble.fsmall.state_dict())
        
        # Freeze first layers
        for p in f_base.parameters():
            p.requires_grad = False
            
        embedding_model = trial_config.get("embedding_model", "conv_gap")
        in_channels = f_base.out_features
        cnn_channels = trial_config.get("cnn_channels", 16)
        extra_conv = trial_config.get("extra_conv", False)
        pool_size = trial_config.get("pool_size", 5)
        hidden_dim = trial_config.get("hidden_dim", 256)
        dropout = trial_config.get("dropout", 0.0)
        pool_type = trial_config.get("pool_type", "max")

        embedding_processing, feature_source = build_embedding_processing(
            embedding_model,
            in_channels=in_channels,
            cnn_channels=cnn_channels,
            extra_conv=extra_conv,
            pool_size=pool_size,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pool_type=pool_type
        )
        embedding_processing = embedding_processing.to(device)
        print(f"Embedding processing: {embedding_model} | feature_source={feature_source}")
        
        rejector = EmbeddingRejector(f_base, embedding_processing, device, feature_source=feature_source)
        
    elif rejector_type == "snrdt":
        print("Initializing SNR DT Rejector (Type 3)...")
        use_meta_snr = trial_config.get("use_meta_snr", False)
        rejector = SNRDT_Rejector(device, use_meta_snr=use_meta_snr)
        
    else:
        raise ValueError(f"Unknown rejector_type: {rejector_type}")
        
    return rejector


def build_run_name(rejector_type, trial_config, worker_id, trial, routing_stage="r1"):
    lr = trial_config["lr"]
    wd = trial_config["wd"]
    dropout = trial_config["dropout"]

    if rejector_type == "snrdt":
        use_meta = trial_config.get("use_meta_snr", False)
        return (
            f"run_{rejector_type}_{routing_stage}"
            f"_metaSNR{use_meta}"
            f"_worker{worker_id}_trial{trial}"
        )

    if rejector_type != "embedding":
        return (
            f"run_{rejector_type}_{routing_stage}"
            f"_lr{lr:.2e}_wd{wd:.2e}_drop{dropout}"
            f"_worker{worker_id}_trial{trial}"
        )

    embedding_model = trial_config["embedding_model"]
    base = (
        f"run_{rejector_type}_{routing_stage}_{embedding_model}"
        f"_lr{lr:.2e}_wd{wd:.2e}_drop{dropout}"
    )
    
    if embedding_model == "resnet_small":
        model_part = (
            f"_channels{trial_config['cnn_channels']}"
            f"_extra{trial_config['extra_conv']}"
        )

    elif embedding_model == "conv_gap":
        model_part = (
            f"_channels{trial_config['cnn_channels']}"
            f"_extra{trial_config['extra_conv']}"
        )
    elif embedding_model == "conv_mlp":
        model_part = (
            f"_channels{trial_config['cnn_channels']}"
            f"_extra{trial_config['extra_conv']}"
            f"_pool{trial_config['pool_size']}"
            f"_hidden{trial_config['hidden_dim']}"
        )
    elif embedding_model == "spatial_pool_mlp":
        model_part = (
            f"_pool{trial_config['pool_size']}"
            f"_ptype{trial_config.get('pool_type', 'max')}"
            f"_hidden{trial_config['hidden_dim']}"
        )
    elif embedding_model == "pooled_mlp":
        model_part = f"_hidden{trial_config['hidden_dim']}"
    elif embedding_model == "pooled_mlp_128_64":
        model_part = "_hidden128_64"
    elif embedding_model == "pooled_mlp_256_256":
        model_part = "_hidden256_256"
    else:
        model_part = ""

    print(f"now running: {base}{model_part}_worker{worker_id}_trial{trial}")
    return f"{base}{model_part}_worker{worker_id}_trial{trial}"


if __name__ == "__main__":
    worker_seed = 42 + args.worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    
    print(f"--- Grid Search Rejector | Worker {args.worker_id} ---")
    print_runtime_diagnostics(_config)
    
    # load data and base ensemble
    ensemble, train_loader, train_targets, val_loader, val_targets, uncut_train_loader, uncut_val_loader = load_data(_config)
    
    # start Trial Loop
    for trial in range(_config["num_trials"]):
        # determine the rejector type
        if args.rejector_type == "random":
            r_type = random.choice(["standard", "embedding", "snrdt"])
        else:
            r_type = args.rejector_type
            
        # Config values override the previous Random-Search defaults.
        configured_lr = _config.get("learning_rate")
        configured_wd = _config.get("weight_decay")
        if configured_lr is not None and configured_lr <= 0:
            raise ValueError("config['learning_rate'] must be greater than 0")
        if configured_wd is not None and configured_wd < 0:
            raise ValueError("config['weight_decay'] must not be negative")

        lr = (
            configured_lr
            if configured_lr is not None
            else 10 ** random.uniform(-6, -4)
        )
        wd = configured_wd if configured_wd is not None else 0.0
        dropout = random.choice(_config.get("dropout_choices", [0.0]))
        embedding_model = random.choice(_config.get("embedding_model_choices", ["conv_gap"]))
        cnn_channels = random.choice(_config.get("cnn_channels_choices", [16, 32, 64, 128]))
        extra_conv = random.choice(_config.get("extra_conv_choices", [False, True]))
        pool_size = random.choice(_config.get("pool_size_choices", [5]))
        pool_type = random.choice(_config.get("pool_type_choices", ["max"]))
        hidden_dim = random.choice(_config.get("hidden_dim_choices", [256]))
        use_meta_snr = random.choice(_config.get("use_meta_snr_choices", [True, False]))

        trial_config = {
            "lr": lr,
            "wd": wd,
            "dropout": dropout,
            "embedding_model": embedding_model,
            "cnn_channels": cnn_channels,
            "extra_conv": extra_conv,
            "pool_size": pool_size,
            "pool_type": pool_type,
            "hidden_dim": hidden_dim,
            "use_meta_snr": use_meta_snr,
        }
        
        print(f"\n[{r_type}] | Worker {args.worker_id} | Trial {trial+1}/{_config['num_trials']}")
        print(f"Hyperparams: {trial_config}")
        
        # Create the rejector
        rejector = get_rejector(r_type, _config, ensemble, trial_config)
        ensemble.rejector = rejector # Swap rejector in the ensemble

        # Setting up Tensorboard Writer
        run_name = build_run_name(r_type, trial_config, args.worker_id, trial, 
                                  routing_stage=_config.get("routing_stage", "r1"))
        log_dir_parts = [_config["tensorboard_root"], r_type]
        if r_type == "embedding":
            log_dir_parts.append(embedding_model)
        writer = SummaryWriter(log_dir=os.path.join(*log_dir_parts, run_name))

        try:
            # Fit the rejector
            print(f"Fitting rejector...")
            ensemble.rejector.fit(
                train_loader, train_targets, val_loader, val_targets,
                lr=lr, weight_decay=wd, 
                num_epochs=_config.get("num_epochs", 100), 
                patience=_config.get("patience", 15),
                writer=writer,
                run_name=run_name
            )
            
            # Evaluate Ensemble
            print(f"Evaluating Ensemble on UNCUT data...")
            #ensemble.eval(train_dataloader=uncut_train_loader, val_dataloader=uncut_val_loader)
            
            writer.close()

            
        except Exception as e:
            print(f"Trial failed with error: {e}")
            
    print("Grid search finished!")
