import math
import numpy as np
import torch

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - optional dependency
    go = None


def _require_plotly():
    if go is None:
        raise ImportError(
            "Plotly is required for visualization helpers. Install plotly to enable these functions."
        )

def plot_snr_distributions(dataset, idx_0, idx_1, figsize=(10, 6), jitter=0.18, seed=0):
    """Boxplot SNR per subset + jitter scatter colored by pulse label (1/0) + NaN/count annotations."""

    _require_plotly()

    rng = np.random.default_rng(seed)

    def _snr_and_labels(indices):
        snr_values = []
        pulse_values = []
        nan_count = 0

        for idx in indices:
            sample = dataset[idx]

            snr = sample["metadata"][0]
            pulse = sample["label"]

            if torch.is_tensor(snr):
                snr = snr.item()
            snr = float(snr)

            if torch.is_tensor(pulse):
                pulse = pulse.item()
            pulse = int(pulse)

            if math.isnan(snr):
                nan_count += 1
            else:
                snr_values.append(snr)
                pulse_values.append(pulse)

        snr_values = np.asarray(snr_values, dtype=float)
        pulse_values = np.asarray(pulse_values, dtype=int)

        pulse_count = int((pulse_values == 1).sum())
        artefact_count = int((pulse_values == 0).sum())
        return snr_values, pulse_values, nan_count, pulse_count, artefact_count

    groups = [
        ("class 0", idx_0),
        ("class 1", idx_1),
        ("all", range(len(dataset))),
    ]

    # Extract data
    group_data = []
    for name, indices in groups:
        snr, pulse, nan_c, pulse_c, art_c = _snr_and_labels(indices)
        group_data.append((name, snr, pulse, nan_c, pulse_c, art_c))

    labels = [
        f"{name} (n={len(snr)})"
        for (name, snr, pulse, nan_c, pulse_c, art_c) in group_data
    ]

    width = int(figsize[0] * 100)
    height = int(figsize[1] * 100)

    fig = go.Figure()

    # --- Boxplots (one per subset) ---
    for i, ((name, snr, pulse, nan_c, pulse_c, art_c), label) in enumerate(zip(group_data, labels)):
        fig.add_trace(
            go.Box(
                x=[i] * len(snr),
                y=snr,
                name=label,
                boxmean=True,
                opacity=0.6,
                line=dict(color="#2f2f2f", width=1.5),
            )
        )

    # --- Scatter overlay: artefact (0) vs pulse (1) ---
    # One trace per class (0/1) for a clean legend
    x_art, y_art = [], []
    x_pulse, y_pulse = [], []

    for i, (name, snr, pulse, nan_c, pulse_c, art_c) in enumerate(group_data):
        if len(snr) == 0:
            continue

        j = rng.uniform(-jitter, jitter, size=len(snr))
        x = i + j

        mask_p = (pulse == 1)
        mask_a = (pulse == 0)

        x_pulse.extend(x[mask_p].tolist())
        y_pulse.extend(snr[mask_p].tolist())

        x_art.extend(x[mask_a].tolist())
        y_art.extend(snr[mask_a].tolist())

    fig.add_trace(
        go.Scatter(
            x=x_art, y=y_art,
            mode="markers",
            name="artefact (label=0)",
            marker=dict(size=6, symbol="circle", opacity=0.55),
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_pulse, y=y_pulse,
            mode="markers",
            name="pulse (label=1)",
            marker=dict(size=7, symbol="diamond", opacity=0.75),
            showlegend=True,
        )
    )

    # --- Layout / axes ---
    fig.update_layout(
        width=width,
        height=height,
        title="SNR distribution per subset (with pulse/artefact overlay)",
        yaxis_title="SNR (metadata[0])",
        template="plotly_white",
        boxmode="group",
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(labels))),
        ticktext=labels,
        tickangle=10,
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0, 0, 0, 0.3)")

    # --- Annotations: NaNs + counts ---
    all_valid = np.concatenate([snr for (_, snr, *_rest) in group_data if len(snr) > 0], axis=0) \
        if any(len(snr) > 0 for (_, snr, *_rest) in group_data) else np.asarray([1.0])
    global_max = float(np.max(all_valid))
    y_anno = global_max + 0.05 * (abs(global_max) + 1.0)

    for i, (name, snr, pulse, nan_c, pulse_c, art_c) in enumerate(group_data):
        fig.add_annotation(
            x=i,
            y=y_anno,
            text=f"NaNs: {nan_c}<br>pulse: {pulse_c} | artefact: {art_c}",
            showarrow=False,
            yanchor="bottom",
            align="center",
            font=dict(size=12, color="#2f2f2f"),
        )

    fig.show()

def eval_optimal(fsmall, fbig, train_dataloader = None, test_dataloader = None, optimal_train_routing = None, optimal_test_routing = None):
    
    def predict_proba(batch, routing, fsmall, fbig):
        labels = batch["label"]
        batch_size = labels.shape[0]

        routing_t = torch.as_tensor(routing, device=labels.device).view(-1)
        if routing_t.numel() != batch_size:
            raise ValueError(
                f"Routing length ({routing_t.numel()}) does not match batch size ({batch_size})."
            )

        device = getattr(fsmall, "device", labels.device)
        routing_t = routing_t.to(device)

        with torch.no_grad():
            small_outputs = fsmall(batch).to(device)
            big_outputs = fbig(batch).to(device)

        outputs = torch.empty_like(small_outputs)
        small_mask = routing_t == 0
        big_mask = routing_t == 1

        if small_mask.any():
            outputs[small_mask] = small_outputs[small_mask]
        if big_mask.any():
            outputs[big_mask] = big_outputs[big_mask]

        # In case routing contains unexpected labels, fall back to the small model
        other_mask = (~small_mask) & (~big_mask)
        if other_mask.any():
            outputs[other_mask] = small_outputs[other_mask]

        return outputs   
        
    train_acc = None
    val_acc = None
    test_acc = None
    
    train_loss = None
    val_loss = None
    test_loss = None
    
    fsmall.eval()
    fbig.eval()
    
    criterion = torch.nn.CrossEntropyLoss()
    if train_dataloader is not None:
        print("evaluating model on training data")
        train_running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        optimal_train_routing_t = torch.as_tensor(optimal_train_routing).view(-1)
        routing_offset = 0
        
        for batch in train_dataloader:
            current_batch_size = batch["label"].shape[0]
            batch_routing = optimal_train_routing_t[routing_offset:routing_offset + current_batch_size]
            if batch_routing.numel() != current_batch_size:
                raise ValueError(
                    "Train routing targets do not align with the dataloader batches."
                )
            routing_offset += current_batch_size
            
            outputs = predict_proba(batch, batch_routing, fsmall, fbig)
            labels = batch["label"].to(fsmall.device)
            
            loss = criterion(outputs.float(), torch.nn.functional.one_hot(labels, num_classes=2).float())
            train_running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        if routing_offset != optimal_train_routing_t.numel():
            raise ValueError(
                f"Unused train routing entries: expected {routing_offset}, got {optimal_train_routing_t.numel()}"
            )

        train_loss = train_running_loss / max(len(train_dataloader), 1)
        train_acc = correct_train / max(total_train, 1)
        
        print(f"train acc: {train_acc}; train loss: {train_loss}")
        
    if test_dataloader is not None:
        print("evaluating model on test data")
        test_running_loss = 0.0
        correct_test = 0
        total_test = 0
        
        optimal_test_routing_t = torch.as_tensor(optimal_test_routing).view(-1)
        routing_offset = 0
        
        for batch in test_dataloader:
            current_batch_size = batch["label"].shape[0]
            batch_routing = optimal_test_routing_t[routing_offset:routing_offset + current_batch_size]
            if batch_routing.numel() != current_batch_size:
                raise ValueError(
                    "Test routing targets do not align with the dataloader batches."
                )
            routing_offset += current_batch_size
            
            outputs = predict_proba(batch, batch_routing, fsmall, fbig)
            labels = batch["label"].to(fsmall.device)
            
            loss = criterion(outputs.float(), torch.nn.functional.one_hot(labels, num_classes=2).float())
            test_running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
        
        if routing_offset != optimal_test_routing_t.numel():
            raise ValueError(
                f"Unused test routing entries: expected {routing_offset}, got {optimal_test_routing_t.numel()}"
            )

        test_loss = test_running_loss / max(len(test_dataloader), 1)
        test_acc = correct_test / max(total_test, 1)
        
        print(f"test acc: {test_acc}; test loss: {test_loss}")
        
    return train_acc, train_loss, val_acc, val_loss, test_acc, test_loss


def _maybe_extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint

def _strip_prefix_from_state_dict(state_dict, prefix="model."):
    if not isinstance(state_dict, dict):
        return state_dict
    if prefix and all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict

def _extract_labels(dataset):
    labels = []
    for sample in dataset:
        label = sample["label"]
        if torch.is_tensor(label):
            label = label.item()
        labels.append(int(label))
    return np.asarray(labels)
    
    
def plot_snr_distributions(dataset, idx_0, idx_1, figsize=(10, 6)):
    """Plot SNR distributions for class-specific subsets and the full dataset."""

    _require_plotly()

    def _snr_values(indices):
        values = []
        for idx in indices:
            sample = dataset[idx]
            snr = sample["metadata"][0]
            if torch.is_tensor(snr):
                snr = snr.item()
            snr = float(snr)
            if math.isnan(snr):
                snr = -1.0
            values.append(snr)
        return values

    snr_class0 = _snr_values(idx_0)
    snr_class1 = _snr_values(idx_1)
    snr_all = _snr_values(range(len(dataset)))

    labels = [
        f"class 0 (n={len(snr_class0)})",
        f"class 1 (n={len(snr_class1)})",
        f"all (n={len(snr_all)})",
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    width = int(figsize[0] * 100)
    height = int(figsize[1] * 100)

    fig = go.Figure()
    for values, label, color in zip([snr_class0, snr_class1, snr_all], labels, colors):
        fig.add_trace(
            go.Box(
                y=values,
                name=label,
                boxmean=True,
                marker_color=color,
                fillcolor=color,
                opacity=0.6,
                line=dict(color="#2f2f2f", width=1.5),
            )
        )

    fig.update_layout(
        width=width,
        height=height,
        title="SNR distribution per subset",
        yaxis_title="SNR (metadata[0])",
        template="plotly_white",
        showlegend=False,
        boxmode="group",
        xaxis=dict(tickangle=10),
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0, 0, 0, 0.3)")
    fig.show()