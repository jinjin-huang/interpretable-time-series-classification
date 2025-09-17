import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
from torch.utils.data import DataLoader

from src.utils.helpers import load_ts_file
from src.utils.settings import (
    dataset_name,
    experiment_run,
    num_classes,
    num_dimensions,
    train_file,
    test_file,
)
from src.utils.TimeSeriesDataset import TimeSeriesDataset
from src.ProtoPConv.convlstm_features import convlstm_features
import os

base_architecture_to_features = {
    "convlstm": convlstm_features,
}


class ProtoLayer(nn.Module):
    def __init__(self, n_proto, proto_channels, proto_len):
        super(ProtoLayer, self).__init__()
        self.prototypes = nn.Parameter(torch.rand(n_proto, proto_channels, proto_len))
        self.n_proto = n_proto
        self.n_channels = proto_channels
        self.proto_len = proto_len

    def forward(self, x):
        ones = torch.ones(self.prototypes.shape, device=x.device)
        x2 = x**2
        x2_patch_sum = F.conv1d(x2, ones)

        p2 = torch.sum(self.prototypes**2, dim=(1, 2))  # (N,)
        p2_reshape = p2.view(1, -1, 1)

        xp = F.conv1d(x, self.prototypes)

        return F.relu(x2_patch_sum - 2 * xp + p2_reshape)


class ProtoPLSTM(nn.Module):
    def __init__(
        self,
        num_classes,
        num_dimensions,
        use_gcb=False,
        num_prototypes=5,
        prototype_shape=1,
        pool_type="max",
    ):
        super(ProtoPLSTM, self).__init__()
        self.features = convlstm_features(num_dimensions=num_dimensions)
        self.use_gcb = use_gcb
        self.gcb = None
        self.num_dimensions = num_dimensions
        self.proto_layer = None
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.prototype_shape = prototype_shape
        self.pool_type = pool_type
        self.fc_layer = nn.Linear(num_classes * num_prototypes, num_classes)
        self.epsilon = 1e-4

    def conv_features(self, x):
        x = self.features(x)
        return x

    def pool(self, x):
        if self.pool_type == "max":
            return F.adaptive_max_pool1d(x, 1).squeeze(-1)  # → (B, C)
        elif self.pool_type == "avg":
            return F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # → (B, C)

    def forward(self, x):
        x = self.conv_features(x)
        if self.proto_layer is None:
            self.proto_layer = ProtoLayer(
                self.num_classes * self.num_prototypes, x.shape[1], self.prototype_shape
            ).to(x.device)
        x = self.proto_layer(x)
        x = self.pool(x)
        x = self.fc_layer(x)
        return x

    def project_prototypes(self, x, y):
        self.proto_info = []
        print("Projecting prototypes...", end="")
        with torch.no_grad():
            x = self.features(x)  # (B, C, T)
            if self.use_gcb:
                x = self.gcb_forward(x)  # (B, C, T)

            proto_out = self.proto_layer(x)  # (B, N, T') ← 1D output

            n_samples, n_proto, t_out = proto_out.shape
            proto_len = self.proto_layer.proto_len  # sliding window length
            for indx_proto in range(n_proto):
                min_dist = np.inf
                if indx_proto == 140:
                    print("15")
                elif indx_proto % 10 == 0:
                    print(f"{int(indx_proto / 10 + 1)}..", end="")

                for indx_sample in range(n_samples):
                    for t in range(t_out):
                        dist = proto_out[indx_sample, indx_proto, t].item()
                        if dist < min_dist:
                            t_min = t
                            indx_sample_min = indx_sample
                            min_dist = dist

                self.proto_info.append(
                    (indx_proto, indx_sample_min, y[indx_sample_min].item())
                )
                best_patch = x[
                    indx_sample_min, :, t_min : t_min + proto_len
                ]  # (C, proto_len)
                self.proto_layer.prototypes.data[indx_proto] = best_patch


def check_gpu_usage():
    if torch.cuda.is_available():
        print(
            f"GPU Usage: {torch.cuda.memory_allocated() / 1e6:.2f} MB allocated / {torch.cuda.memory_reserved() / 1e6:.2f} MB reserved"
        )
        print(f"CUDA Memory Cached: {torch.cuda.memory_cached() / 1e6:.2f} MB")
    else:
        print("CUDA is not available. Running on CPU.")


def consturct_ProPLSTM(base_architecture, num_classes, num_dimensions, use_gcb=False):
    features = base_architecture_to_features[base_architecture]
    model = ProtoPLSTM(
        num_classes=num_classes, num_dimensions=num_dimensions, use_gcb=use_gcb
    )
    return model


def train(model, train_loader: DataLoader, epoch, device, optimizer, criterion):
    model.train()
    check_gpu_usage()
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
    check_gpu_usage()
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
    check_gpu_usage()
    return correct / len(test_loader.dataset)


import time
from torch.utils.data import random_split

if __name__ == "__main__":
    print("loading datasets")
    X_train, y_train = load_ts_file(train_file)
    full_train_dataset = TimeSeriesDataset(X_train, y_train)

    val_ratio = 0.2
    val_size = int(len(full_train_dataset) * val_ratio)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=False
    )

    X_test, y_test = load_ts_file(test_file)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=False
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = consturct_ProPLSTM(
        base_architecture="convlstm",
        num_classes=num_classes,
        num_dimensions=num_dimensions,
        use_gcb=False,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    r = 5
    best_val_acc = 0
    best_test_acc_at_val = 0

    total_start_time = time.time()

    for epoch in range(100):
        epoch_start_time = time.time()

        train(model, train_loader, epoch, device, optimizer, criterion)

        projection = not divmod(epoch, r)[-1] and epoch != 0
        if projection:
            x_all = torch.stack(
                [full_train_dataset[i][0] for i in train_dataset.indices]
            ).to(device)
            y_all = torch.tensor(
                [full_train_dataset[i][1] for i in train_dataset.indices],
                dtype=torch.long,
            ).to(device)

            model.project_prototypes(x_all, y_all)

        val_acc = test(model, val_loader, device, criterion, scheduler)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = test(model, test_loader, device, criterion, scheduler)
            best_test_acc_at_val = test_acc
            print(
                f"New best val_acc: {best_val_acc:.4f}, corresponding test_acc: {test_acc:.4f}"
            )

        epoch_end_time = time.time()
        print(
            f"Epoch {epoch} finished in {(epoch_end_time - epoch_start_time):.2f} sec"
        )

    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time

    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy at Best Validation Epoch: {best_test_acc_at_val:.4f}")
    print(f"Total training time: {total_training_time:.2f} sec")

    results_file = "results/protopconv/protopconv_accuracy.csv"
    file_exists = os.path.isfile(results_file)

    with open(results_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "dataset_name",
                    "experiment_run",
                    "best_val_acc",
                    "best_test_acc_at_val",
                    "training_time_sec",
                ]
            )
        writer.writerow(
            [
                dataset_name,
                experiment_run,
                f"{best_val_acc:.4f}",
                f"{best_test_acc_at_val:.4f}",
                f"{total_training_time:.2f}",
            ]
        )
