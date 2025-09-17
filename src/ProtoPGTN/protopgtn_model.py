import csv
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.helpers import load_ts_file
import os
from src.utils.settings import (
    train_file,
    test_file,
    dataset_name,
    experiment_run,
    num_classes,
    num_dimensions,
    data_length,
    args,
)
from src.utils.TimeSeriesDataset import TimeSeriesDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from src.ProtoPGTN.gated_transformer import Gated_Transformer
from sklearn.model_selection import KFold
import random
import time
from sklearn.model_selection import train_test_split


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ProtoLayer(nn.Module):
    def __init__(self, n_proto, proto_dim):
        super().__init__()
        # Learnable prototype vectors of shape (n_proto, proto_dim)
        self.prototypes = nn.Parameter(torch.randn(n_proto, proto_dim))
        self.n_proto = n_proto
        self.proto_dim = proto_dim
        self.epsilon = 1e-4

    def forward(self, x):
        # cosine similarity

        x_norm = F.normalize(x, p=2, dim=1)
        p_norm = F.normalize(self.prototypes, p=2, dim=1)
        # Compute cosine similarity between inputs and prototypes
        sim_scores = torch.mm(x_norm, p_norm.t())
        # return distance as 1 - cosine similarity
        return 1 - sim_scores

        # # Euclidean distance
        # dist = torch.cdist(x, self.prototypes, p=2)  # shape: (batch, n_proto)
        # return dist


class ProtoPGTN(nn.Module):
    def __init__(self, num_classes, num_dimensions, num_prototypes=5, feature_dim=128):
        super(ProtoPGTN, self).__init__()
        # Gated Transformer for feature extraction
        self.features = Gated_Transformer(
            d_model=args.d_model,
            d_input=data_length,
            d_channel=num_dimensions,
            d_hidden=args.d_hidden,
            q=args.q,
            v=args.v,
            h=args.h,
            N=args.N,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dropout=args.dropout,
            pe=True,
            mask=True,
            d_output=num_classes,
        )
        # Feature adapter to map to feature_dim
        self.feature_adapter = nn.Sequential(nn.LazyLinear(feature_dim), nn.ReLU())
        self.proto_layer = ProtoLayer(num_classes * num_prototypes, feature_dim)
        self.fc_layer = nn.Linear(num_classes * num_prototypes, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        encoding = self.features(x, stage="train" if self.training else "test")
        encoding = self.feature_adapter(encoding)
        dist = self.proto_layer(encoding)
        sim_scores = torch.log10((dist + 1) / (dist + self.proto_layer.epsilon))
        output = self.fc_layer(sim_scores)
        return output

    def project_prototypes(self, x, y):
        """Project prototypes to the nearest training sample features."""
        x = x.transpose(1, 2)
        with torch.no_grad():
            encodings = self.features(x, stage="train" if self.training else "test")
            encodings = self.feature_adapter(encodings)
            encodings = F.normalize(encodings, p=2, dim=1)

            prototypes = F.normalize(self.proto_layer.prototypes, p=2, dim=1)

            self.proto_info = []
            print("Projecting prototypes (cosine)...", end="")

            sim_matrix = torch.mm(encodings, prototypes.t())

            for proto_id in range(self.proto_layer.n_proto):
                max_sim, max_idx = torch.max(sim_matrix[:, proto_id], dim=0)

                raw_feature = self.feature_adapter(self.features(x, stage="train"))[
                    max_idx
                ]
                self.proto_layer.prototypes.data[proto_id] = raw_feature

                self.proto_info.append(
                    (
                        proto_id,
                        max_idx.item(),
                        y[max_idx].item(),
                        max_sim.item(),
                    )
                )

                if proto_id % 10 == 0:
                    print(f"{int(proto_id/10)+1}..", end="")

        print(f"\nProjected {self.proto_layer.n_proto} prototypes")
        print(f"Average max similarity: {np.mean([x[3] for x in self.proto_info]):.4f}")

    def project_prototypes_batched(self, x, y, batch_size=32):
        """Project prototypes to the nearest training sample features in batches.
        This is more memory efficient for large datasets."""
        x = x.transpose(1, 2)
        with torch.no_grad():
            prototypes = F.normalize(self.proto_layer.prototypes, p=2, dim=1)
            n_proto = self.proto_layer.n_proto

            max_sims = torch.full((n_proto,), -float("inf"), device=x.device)
            max_indices = torch.full((n_proto,), -1, dtype=torch.long, device=x.device)

            print("Projecting prototypes (cosine, batched)...")

            num_samples = x.size(0)
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                x_batch = x[start:end]
                y_batch = y[start:end]

                enc = self.feature_adapter(
                    self.features(x_batch, stage="train" if self.training else "test")
                )
                enc = F.normalize(enc, p=2, dim=1)

                sim_matrix = torch.mm(enc, prototypes.t())
                for proto_id in range(n_proto):
                    batch_max_sim, batch_max_idx = torch.max(
                        sim_matrix[:, proto_id], dim=0
                    )
                    if batch_max_sim > max_sims[proto_id]:
                        max_sims[proto_id] = batch_max_sim
                        max_indices[proto_id] = start + batch_max_idx  # 全局索引

                print(f"{min(end, num_samples)}/{num_samples}", end="\r")

            self.proto_info = []
            for proto_id in range(n_proto):
                max_idx = max_indices[proto_id].item()
                max_sim = max_sims[proto_id].item()
                raw_feature = self.feature_adapter(
                    self.features(x[[max_idx]], stage="train")
                ).squeeze(0)
                self.proto_layer.prototypes.data[proto_id] = raw_feature

                self.proto_info.append(
                    (
                        proto_id,
                        max_idx,
                        y[max_idx].item(),
                        max_sim,
                    )
                )

            print(f"\nProjected {n_proto} prototypes")
            print(
                f"Average max similarity: {np.mean([x[3] for x in self.proto_info]):.4f}"
            )


def check_gpu_usage():
    if torch.cuda.is_available():
        print(
            f"GPU Usage: {torch.cuda.memory_allocated() / 1e6:.2f} MB allocated / {torch.cuda.memory_reserved() / 1e6:.2f} MB reserved"
        )
        print(f"CUDA Memory Cached: {torch.cuda.memory_cached() / 1e6:.2f} MB")
    else:
        print("CUDA is not available. Running on CPU.")


def construct_ProtoPGTN(num_classes, num_dimensions):
    model = ProtoPGTN(num_classes=num_classes, num_dimensions=num_dimensions)
    return model


def train(model, train_loader: DataLoader, epoch, device, optimizer, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).float(), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print(
        "Train Epoch: {} \tLoss: {:.6f}".format(
            epoch, loss.item() / train_loader.batch_size
        )
    )
    check_gpu_usage()


def test(model, test_loader, device, criterion, scheduler):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).float(), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    scheduler.step(test_loss)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)


def test_tuned_model():
    set_seed(args.random_seed)
    print(f"Running with random seed: {args.random_seed}")

    print("loading datasets")
    X_train_full, y_train_full = load_ts_file(train_file)
    # split training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=args.random_seed,
    )

    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    X_test, y_test = load_ts_file(test_file)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    # initialize model, optimizer, scheduler, criterion
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = construct_ProtoPGTN(
        num_classes=num_classes, num_dimensions=num_dimensions
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    r = 5
    best_val_acc = 0
    test_acc_at_best_val = 0

    save_name = (
        Path(__file__).resolve().parent
        / f"../../saved_models/protopgtn/best_models/{dataset_name}/{experiment_run}.pt"
    )
    proto_info_name = (
        Path(__file__).resolve().parent
        / f"../../saved_models/protopgtn/prototypes/{dataset_name}/{experiment_run}.csv"
    )

    save_name.parent.mkdir(parents=True, exist_ok=True)
    proto_info_name.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for epoch in range(args.num_epochs):
        train(model, train_loader, epoch, device, optimizer, criterion)

        projection = not divmod(epoch, r)[-1] and epoch != 0
        if projection:
            x_all = torch.Tensor(train_dataset[:][0]).to(device)
            y_all = torch.LongTensor(train_dataset[:][1]).to(device)
            model.project_prototypes_batched(x_all, y_all)

        val_acc = test(model, val_loader, device, criterion, scheduler)

        # use validation accuracy to save the best model
        if val_acc > best_val_acc and projection:
            best_val_acc = val_acc
            test_acc = test(model, test_loader, device, criterion, scheduler)
            test_acc_at_best_val = test_acc
            print(
                f"New best validation acc: {best_val_acc:.4f}, Test acc: {test_acc:.4f}"
            )
            torch.save(model.state_dict(), save_name)
            with open(proto_info_name, "w", newline="") as csvfile:
                protowriter = csv.writer(
                    csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
                )
                protowriter.writerow(
                    ["prototype", "train_sample", "label", "similarity"]
                )
                protowriter.writerows(model.proto_info)

    elapsed_time = time.time() - start_time
    print(f"Training time for {dataset_name}: {elapsed_time:.2f} seconds")

    # save training results
    result_file = Path("results/protopgtn/training_results_protopgtn.csv")
    file_exists = result_file.exists()

    with open(result_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "Dataset",
                    "Experiment",
                    "Training Time (s)",
                    "Best Val Acc",
                    "Test Acc at Best Val",
                ]
            )
        writer.writerow(
            [
                dataset_name,
                experiment_run,
                elapsed_time,
                best_val_acc,
                test_acc_at_best_val,
            ]
        )


def k_fold_cross_validation(k, model_class, train_data, device, criterion, args):
    X_train, y_train = train_data
    if isinstance(X_train, list):
        X_train = np.array(X_train)
    if isinstance(y_train, list):
        y_train = np.array(y_train)

    kf = KFold(n_splits=k, shuffle=True, random_state=args.random_seed)
    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\n===== Fold {fold + 1} / {k} =====")

        X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
        X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]

        train_dataset = TimeSeriesDataset(X_train_fold, y_train_fold)
        val_dataset = TimeSeriesDataset(X_val_fold, y_val_fold)

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        model = model_class(num_classes=num_classes, num_dimensions=num_dimensions)
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)

        best_acc = 0
        r = 5
        for epoch in range(args.num_epochs):
            train(model, train_loader, epoch, device, optimizer, criterion)

            projection = not divmod(epoch, r)[-1] and epoch != 0
            if projection:
                x_all = torch.Tensor(train_dataset[:][0]).to(device)
                y_all = torch.LongTensor(train_dataset[:][1]).to(device)
                model.project_prototypes(x_all, y_all)

            val_acc = test(model, val_loader, device, criterion, scheduler)
            if val_acc > best_acc:
                best_acc = val_acc
                print(f"Fold {fold + 1} Current Best Accuracy: {best_acc}")

        fold_accs.append(best_acc)

    avg_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    print(f"\n==== {k}-Fold Results ====")
    print(f"Average Accuracy: {avg_acc:.4f}")
    print(f"Standard Deviation: {std_acc:.4f}")
    return avg_acc, std_acc


def hyperparameter_tuning():
    all_results = []
    d_model_values = [64]
    random_seeds = [0, 21]

    for d_model in d_model_values:
        for seed in random_seeds:
            print(
                f"\n===== Parameter Combination: d_model={d_model}, random_seed={seed} ====="
            )
            args.d_model = d_model
            args.random_seed = seed
            set_seed(seed)

            X_train, y_train = load_ts_file(train_file)
            train_data = (X_train, y_train)

            avg_acc, std_acc = k_fold_cross_validation(
                k=5,
                model_class=ProtoPGTN,
                train_data=train_data,
                device="cuda" if torch.cuda.is_available() else "cpu",
                criterion=nn.CrossEntropyLoss(),
                args=args,
            )

            all_results.append(
                {
                    "dataset_name": dataset_name,
                    "d_model": d_model,
                    "random_seed": seed,
                    "avg_acc": avg_acc,
                    "std_acc": std_acc,
                }
            )

    print("\n===== Hyperparameter Tuning Results =====")
    for r in all_results:
        print(
            f"dataset={r['dataset_name']}, d_model={r['d_model']}, seed={r['random_seed']}, Average Accuracy={r['avg_acc']:.4f}, Standard Deviation={r['std_acc']:.4f}"
        )

    results_file = "results/protopgtn/hyperparameter_tuning_results.csv"
    write_header = not os.path.exists(results_file)

    with open(results_file, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset_name", "d_model", "random_seed", "avg_acc", "std_acc"],
        )
        if write_header:
            writer.writeheader()
        writer.writerows(all_results)


if __name__ == "__main__":
    if not args.hyperparameter_tuning:
        test_tuned_model()
    else:
        hyperparameter_tuning()
