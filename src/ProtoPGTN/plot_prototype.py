from src.utils.settings import (
    train_file,
    dataset_name,
    experiment_run,
    num_classes,
    num_dimensions,
    data_length,
)
from src.utils.helpers import load_ts_file
from src.utils.TimeSeriesDataset import TimeSeriesDataset
from src.ProtoPGTN.protopgtn_model import construct_ProtoPGTN

import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import math


def load_data(train_file):
    X_train, y_train = load_ts_file(train_file)
    return X_train, y_train


def initialize_model(device):
    model = construct_ProtoPGTN(
        num_classes=num_classes,
        num_dimensions=num_dimensions,
    ).to(device)

    save_name = (
        Path(__file__).resolve().parent
        / f"../../saved_models/protopgtn/best_models/{dataset_name}/{experiment_run}.pt"
    )
    model.load_state_dict(torch.load(save_name, map_location=device))
    model.eval()
    return model


def get_proto_info():
    proto_info_path = (
        Path(__file__).resolve().parent
        / f"../../saved_models/protopgtn/prototypes/{dataset_name}/{experiment_run}.csv"
    )
    return pd.read_csv(proto_info_path, sep=" ")


# generate sliding windows, return list of tensors and their (start, end) positions
def sliding_windows_seq_first(sample, window_size=30, stride=5):
    T, _ = sample.shape
    windows, positions = [], []
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        windows.append(sample[start:end, :])
        positions.append((start, end))
    return windows, positions


# pad to sequence length
def pad_to_sequence_length(windows, total_length=144):
    padded_windows = []
    for win in windows:
        L = win.shape[0]
        pad_len = total_length - L
        if pad_len > 0:
            pad_tensor = torch.zeros(pad_len, win.shape[1], device=win.device)
            padded = torch.cat([win, pad_tensor], dim=0)
        else:
            padded = win
        padded_windows.append(padded)
    return torch.stack(padded_windows)


def find_best_matching_window(model, X_train, proto_info_df, target_proto_id, device):
    entry = proto_info_df[proto_info_df["prototype"] == target_proto_id].iloc[0]
    sample_idx = int(entry["train_sample"])

    sample_np = X_train[sample_idx]
    sample = torch.tensor(sample_np).float().transpose(0, 1).to(device)

    windows, positions = sliding_windows_seq_first(sample, window_size=30, stride=5)
    windows = [w.to(device) for w in windows]
    batch = pad_to_sequence_length(windows, total_length=data_length)

    with torch.no_grad():
        features = model.features(batch, stage="test")
        features = model.feature_adapter(features)
        features = F.normalize(features, p=2, dim=1)

        prototype_vector = model.proto_layer.prototypes[target_proto_id].to(device)
        prototype_vector = F.normalize(prototype_vector.unsqueeze(0), p=2, dim=1)

        sims = F.cosine_similarity(features, prototype_vector)
        best_idx = torch.argmax(sims).item()
        best_start, best_end = positions[best_idx]
        best_sim = sims[best_idx].item()
        best_window = sample[best_start:best_end].transpose(0, 1)  # [9, L]

    return sample, best_start, best_end, best_sim, best_window, sample_idx


def calculate_channel_similarity(best_window, prototype_vector, model, device):
    C, _ = best_window.shape
    proto = F.normalize(prototype_vector.unsqueeze(0), dim=1)
    sims = []

    for c in range(C):
        window_c = torch.zeros_like(best_window, device=device)
        window_c[c] = best_window[c]
        inp = window_c.transpose(0, 1)

        pad_len = 144 - inp.shape[0]
        if pad_len > 0:
            pad_tensor = torch.zeros(pad_len, inp.shape[1], device=device)
            inp = torch.cat([inp, pad_tensor], dim=0)
        inp = inp.unsqueeze(0)

        with torch.no_grad():
            feat = model.features(inp, stage="test")
            feat = model.feature_adapter(feat)
            feat = F.normalize(feat, dim=1)
            sim = F.cosine_similarity(feat, proto.squeeze(0), dim=1).item()
        sims.append(sim)

    best_chan = int(torch.tensor(sims).argmax().item())
    return best_chan, sims


def plot_matched_segment(sample, best_start, best_end, best_chan, target_proto_id):
    segment = sample[best_start:best_end, best_chan].cpu()
    plt.figure(figsize=(8, 4))
    plt.plot(range(best_start, best_end), segment, label="Matched Segment", color="b")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.title(f"Prototype {target_proto_id} - Channel {best_chan}")
    plt.legend()
    plt.tight_layout()

    out_path = Path("results/protopgtn/figures")
    out_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path / f"prototype_{target_proto_id}.png")
    plt.close()


def plot_prototype_by_id(target_proto_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, _ = load_data(train_file)
    model = initialize_model(device)
    proto_info_df = get_proto_info()

    sample, best_start, best_end, best_sim, best_window, _ = find_best_matching_window(
        model, X_train, proto_info_df, proto_id, device
    )

    print(f"Most similar window: [{best_start}:{best_end}], Similarity: {best_sim:.4f}")
    print(f"Most similar window shape: {best_window.shape}")

    prototype_vector = model.proto_layer.prototypes[target_proto_id].to(device)
    prototype_vector = F.normalize(prototype_vector.unsqueeze(0), p=2, dim=1)

    best_chan, sims = calculate_channel_similarity(
        best_window, prototype_vector, model, device
    )
    print(f"Most similar channel: {best_chan}, Similarity: {sims[best_chan]:.4f}")

    plot_matched_segment(sample, best_start, best_end, best_chan, target_proto_id)


def plot_prototypes_grouped_by_class(
    proto_info_df,
    X_train,
    model,
    device,
    output_path=f"results/protopgtn/figures/prototypes/{dataset_name}_prototypes_by_class.png",
):

    proto_info_df["prototype"] = proto_info_df["prototype"].astype(int)
    proto_info_df["label"] = proto_info_df["label"].astype(int)

    grouped = proto_info_df.groupby("label")

    num_classes = len(grouped)
    max_protos_in_class = grouped.size().max()

    _, axes = plt.subplots(
        num_classes,
        max_protos_in_class,
        figsize=(max_protos_in_class * 4, num_classes * 3),
    )

    if num_classes == 1:
        axes = np.expand_dims(axes, axis=0)
    if max_protos_in_class == 1:
        axes = np.expand_dims(axes, axis=1)

    for row_idx, (label, group) in enumerate(grouped):
        proto_ids = group["prototype"].tolist()

        for col_idx in range(max_protos_in_class):
            ax = axes[row_idx, col_idx]

            if col_idx >= len(proto_ids):
                ax.axis("off")
                continue

            proto_id = int(proto_ids[col_idx])

            try:
                sample, best_start, best_end, _, best_window, sample_idx = (
                    find_best_matching_window(
                        model, X_train, proto_info_df, proto_id, device
                    )
                )

                prototype_vector = model.proto_layer.prototypes[proto_id].to(device)
                prototype_vector = F.normalize(
                    prototype_vector.unsqueeze(0), p=2, dim=1
                )

                best_chan, _ = calculate_channel_similarity(
                    best_window, prototype_vector, model, device
                )

                segment = sample[best_start:best_end, best_chan].cpu()

                ax.plot(
                    range(best_start, best_end),
                    segment,
                    label=f"Sample {sample_idx} | C{best_chan} | [{best_start}:{best_end}]",
                )

                ax.set_title(f"Proto {proto_id}", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.legend(fontsize=6)

                if col_idx == 0:
                    ax.set_ylabel(
                        f"Label {label}",
                        fontsize=12,
                        rotation=0,
                        labelpad=40,
                        va="center",
                    )

            except Exception as e:
                ax.set_title(f"Proto {proto_id} Error")
                ax.axis("off")
                print(f"⚠️ Error plotting prototype {proto_id}: {e}")
    plt.suptitle("Prototypes Grouped by Class", fontsize=16, y=1.02)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"✅ All prototypes grouped by class saved to: {output_path}")
    plt.close()


def plot_prototypes():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, _ = load_data(train_file)
    model = initialize_model(device)
    proto_info_df = get_proto_info()

    plot_prototypes_grouped_by_class(proto_info_df, X_train, model, device)


if __name__ == "__main__":
    plot_prototypes()
