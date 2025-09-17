from src.utils.settings import (
    test_file,
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
from pathlib import Path
import matplotlib.pyplot as plt


X_test, y_test = load_ts_file(test_file)
test_dataset = TimeSeriesDataset(X_test, y_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = construct_ProtoPGTN(
    num_classes=num_classes,
    num_dimensions=num_dimensions,
)
model = model.to(device)

save_name = (
    Path(__file__).resolve().parent
    / f"../../saved_models/protopgtn/best_models/{dataset_name}/{experiment_run}.pt"
)
model.load_state_dict(torch.load(save_name, map_location=device))
model.eval()

proto_info_path = (
    Path(__file__).resolve().parent
    / f"../../saved_models/protopgtn/prototypes/{dataset_name}/{experiment_run}.csv"
)
proto_info_df = pd.read_csv(proto_info_path, sep=" ")

# get and normalize prototypes
prototypes = model.proto_layer.prototypes.data
prototypes_norm = F.normalize(prototypes, p=2, dim=1)
print(
    f"Loaded and normalized {prototypes_norm.shape[0]} prototypes, dim = {prototypes_norm.shape[1]}"
)


def extract_features(sample_np, model, device):
    sample = torch.tensor(sample_np).float().to(device)
    sample = sample.transpose(0, 1).unsqueeze(0)
    with torch.no_grad():
        feat = model.features(sample, stage="test")
        feat = model.feature_adapter(feat)
        feat = F.normalize(feat, dim=1)
    return feat


def topk_prototypes(feat, prototypes_norm, k=10):
    sims = torch.mm(feat, prototypes_norm.t()).squeeze(0)
    scores, idxs = sims.topk(k, largest=True)
    return scores, idxs


i = 0
sample_np = X_test[i]
feat_full = extract_features(sample_np, model, device)

# Top-k prototypes
topk = 10
scores, proto_idxs = topk_prototypes(feat_full, prototypes_norm, topk)
print(f"Test sample #{i} to prototypes Top {topk}:")
for rank, (p_idx, score) in enumerate(zip(proto_idxs.tolist(), scores.tolist()), 1):
    print(f"  {rank}. Proto {p_idx:>3d}  sim = {score:.4f}")

best_p = int(proto_idxs[0].item())
best_score = float(scores[0].item())
print(f"\nMost similar prototype = {best_p}, Similarity = {best_score:.4f}")


def sliding_windows_seq_first(sample_ts, window_size=30, stride=5):
    T, _ = sample_ts.shape
    windows, positions = [], []
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        windows.append(sample_ts[start:end, :])
        positions.append((start, end))
    return windows, positions


def pad_to_length(windows, total_length=data_length):
    padded = []
    for win in windows:
        L = win.shape[0]
        pad_len = total_length - L
        if pad_len > 0:
            pad = torch.zeros(pad_len, win.shape[1], device=win.device)
            win = torch.cat([win, pad], dim=0)
        padded.append(win)
    return torch.stack(padded)


sample_ts = torch.tensor(sample_np).float().to(device).transpose(0, 1)
windows, positions = sliding_windows_seq_first(sample_ts)
batch = pad_to_length(windows).to(device)

with torch.no_grad():
    feats = model.features(batch, stage="test")
    feats = model.feature_adapter(feats)
    feats = F.normalize(feats, p=2, dim=1)

# find best matches to all prototypes
best_matches = {}
P = prototypes_norm.shape[0]
for p in range(P):
    proto_vec = prototypes_norm[p].unsqueeze(0).to(device)
    sims = F.cosine_similarity(feats, proto_vec, dim=1)
    best_w = torch.argmax(sims).item()
    start, end = positions[best_w]
    best_matches[p] = (best_w, start, end, sims[best_w].item())

print(f"\nTest sample #{i} to all prototypes best matches:")
for p in range(P):
    w, s, e, sim = best_matches[p]
    print(f"  Proto {p:>3d}: window_idx={w:>2d}, time=[{s:>3d}:{e:>3d}], sim={sim:.4f}")

# calculate channel similarity for best windows
dim_matches = {}
for p, (w, s, e, _) in best_matches.items():
    win = windows[w].transpose(0, 1).to(device)
    proto_vec = prototypes_norm[p].unsqueeze(0).to(device)
    sims_chan = []
    for c in range(win.shape[0]):
        wc = torch.zeros_like(win)
        wc[c] = win[c]
        inp = pad_to_length([wc.transpose(0, 1)])[0].unsqueeze(0)
        with torch.no_grad():
            f = model.features(inp, stage="test")
            f = model.feature_adapter(f)
            f = F.normalize(f, dim=1)
            sims_chan.append(F.cosine_similarity(f, proto_vec, dim=1).item())
    best_chan = int(torch.tensor(sims_chan).argmax().item())
    dim_matches[p] = (best_chan, sims_chan[best_chan])

print(f"\nTest sample #{i} to all prototypes best channels:")
for p in range(P):
    w, s, e, sim_win = best_matches[p]
    chan, sim_chan = dim_matches[p]
    print(
        f"  Proto {p:>3d}: window [{s:>3d}:{e:>3d}], win_sim={sim_win:.4f} → best_chan={chan}, chan_sim={sim_chan:.4f}"
    )


proto_to_class = dict(zip(proto_info_df["prototype"], proto_info_df["label"]))
class_to_protos = {}
for proto_idx, class_id in proto_to_class.items():
    class_to_protos.setdefault(class_id, []).append(proto_idx)

num_classes = len(class_to_protos)
max_protos_per_class = max(len(v) for v in class_to_protos.values())

fig, axes = plt.subplots(
    num_classes,
    max_protos_per_class,
    figsize=(max_protos_per_class * 2.5, num_classes * 2),
    squeeze=False,
)

topk = P
sims_protos = torch.mm(feat_full, prototypes_norm.t()).squeeze(0)
proto_similarity = {
    proto_idx: score for proto_idx, score in zip(range(P), sims_protos.tolist())
}

for row_idx, (class_id, proto_list) in enumerate(sorted(class_to_protos.items())):
    for col_idx, proto_idx in enumerate(proto_list):
        ax = axes[row_idx][col_idx]
        if proto_idx not in best_matches or proto_idx not in dim_matches:
            ax.axis("off")
            continue
        w_idx, start, end, _ = best_matches[proto_idx]
        chan, _ = dim_matches[proto_idx]
        full_channel = sample_ts[:, chan].cpu().numpy()
        seg_start, seg_end = start, end
        ax.plot(range(len(full_channel)), full_channel, color="blue", linewidth=1)
        ax.add_patch(
            plt.Rectangle(
                (seg_start, ax.get_ylim()[0]),
                seg_end - seg_start,
                ax.get_ylim()[1] - ax.get_ylim()[0],
                color="red",
                fill=False,
                linewidth=2,
            )
        )
        ax.set_xlim(0, len(full_channel))
        similarity = proto_similarity.get(proto_idx, 0.0)
        ax.set_title(f"P{proto_idx} (sim: {similarity:.4f})", fontsize=8)
        ax.tick_params(labelsize=6)
    for col_idx in range(len(proto_list), max_protos_per_class):
        axes[row_idx][col_idx].axis("off")
    axes[row_idx][0].set_ylabel(f"Class {class_id}", fontsize=10)

label = y_test[i]
fig.suptitle(f"Sample {i} - Label {label}", fontsize=16, y=0.95)
plt.tight_layout(rect=[0, 0, 1, 0.95])

output_path = (
    Path(__file__).resolve().parent.parent.parent
    / f"results/protopgtn/figures/testing/{dataset_name}/{i}.png"
)
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300)
plt.close()
print(f"Prototype visualization saved to: {output_path}")
