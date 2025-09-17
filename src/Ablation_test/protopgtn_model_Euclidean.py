import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader

from src.ProtoPGTN.gated_transformer import Gated_Transformer
from src.utils.helpers import load_ts_file
from src.utils.settings import (
    args,
    data_length,
    dataset_name,
    experiment_run,
    num_classes,
    num_dimensions,
    test_file,
    train_file,
)
from src.utils.TimeSeriesDataset import TimeSeriesDataset


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
        # Euclidean distance
        dist = torch.cdist(x, self.prototypes, p=2)  # shape: (batch, n_proto)
        return dist


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

    elapsed_time = time.time() - start_time
    print(f"Training time for {dataset_name}: {elapsed_time:.2f} seconds")

    # save training results
    result_file = Path("results/Ablation_test/protopgtn_Euclidean_accuracy.csv")
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


if __name__ == "__main__":
    test_tuned_model()
