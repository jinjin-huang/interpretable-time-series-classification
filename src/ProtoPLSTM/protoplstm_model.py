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
from src.ProtoPLSTM.convlstm_features import convlstm_features


base_architecture_to_features = {
    "convlstm": convlstm_features,
}


# context module
class GCB(nn.Module):
    mean = 0.5
    std = 0.1

    def __init__(self, dim=0):
        super(GCB, self).__init__()

        exp_dim = int(dim * 1.0)

        self.cm = nn.Linear(dim, 1)
        self.wv1 = nn.Linear(dim, exp_dim)
        self.norm = nn.LayerNorm(exp_dim)
        self.gelu = nn.GELU()
        self.wv2 = nn.Linear(exp_dim, dim)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, patches):
        h = patches
        x = self.cm(patches)
        x = torch.bmm(h.permute(0, 2, 1), F.softmax(x, 1)).squeeze(-1)
        x = self.wv1(x)
        x = self.gelu(self.norm(x))
        x = self.wv2(x)
        x = h + x.unsqueeze(1)
        x = self.ffn_norm(x)
        x = torch.sigmoid(x)
        return x


class ProtoLayer(nn.Module):
    def __init__(self, n_proto, proto_channels, proto_h, proto_w):
        super(ProtoLayer, self).__init__()
        self.prototypes = nn.Parameter(
            torch.rand(n_proto, proto_channels, proto_h, proto_w)
        )
        self.n_proto = n_proto
        self.n_channels = proto_channels
        self.hp = proto_h
        self.wp = proto_w

    def forward(self, x):
        ones = torch.ones(self.prototypes.shape, device=x.device)
        x2 = x**2
        x2_patch_sum = F.conv2d(x2, ones)

        p2 = self.prototypes**2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(x, self.prototypes)

        return F.relu(x2_patch_sum - 2 * xp + p2_reshape)


class ProtoPLSTM(nn.Module):
    def __init__(
        self,
        num_classes,
        num_dimensions,
        use_gcb=True,
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

    def gcb_forward(self, x):
        _, _, H, W = x.shape
        gcb_dim = H * W
        if self.gcb is None:
            self.gcb = GCB(gcb_dim).to(x.device)
        x = x.reshape(x.size(0), x.size(1), -1)
        x = self.gcb(x)
        x = x.view(x.size(0), x.size(1), self.num_dimensions, -1)
        return x

    def pool(self, x):
        if self.pool_type == "max":
            return (-F.max_pool2d(-x, (x.shape[-2], x.shape[-1]))).view(-1, x.shape[1])
        elif self.pool_type == "avg":
            return F.adaptive_avg_pool2d(x, 1).view(-1, x.shape[1])

    def forward(self, x):
        x = self.conv_features(x)
        if self.use_gcb:
            x = self.gcb_forward(x)
        if self.proto_layer is None:
            self.proto_layer = ProtoLayer(
                self.num_classes * self.num_prototypes,
                x.shape[1],
                self.prototype_shape,
                self.prototype_shape,
            ).to(x.device)
        x = self.proto_layer(x)
        x = self.pool(x)
        x = self.fc_layer(x)
        return x

    def project_prototypes(self, x, y):
        self.proto_info = []
        print("Projecting prototypes...", end="")
        with torch.no_grad():
            x = self.features(x)
            if self.use_gcb:
                x = self.gcb_forward(x)
            proto_out = self.proto_layer(x)

            n_samples, n_proto, ho, wo = proto_out.shape

            for indx_proto in range(n_proto):
                min_dist = np.inf
                if indx_proto == 140:
                    print(f"15")
                elif indx_proto % 10 == 0:
                    print(f"{int(indx_proto/10+1)}..", end="")
                for indx_sample in range(n_samples):
                    for h in range(ho):
                        for w in range(wo):
                            if (
                                proto_out[indx_sample, indx_proto, h, w].item()
                                < min_dist
                            ):
                                h_min = h
                                w_min = w
                                indx_sample_min = indx_sample
                                min_dist = proto_out[
                                    indx_sample, indx_proto, h, w
                                ].item()
                hp, wp = (
                    self.proto_layer.prototypes.shape[-2],
                    self.proto_layer.prototypes.shape[-1],
                )
                self.proto_info.append(
                    (indx_proto, indx_sample_min, y[indx_sample_min].item())
                )
                self.proto_layer.prototypes.data[indx_proto] = x[
                    indx_sample_min, :, h_min : h_min + hp, w_min : w_min + wp
                ]


def check_gpu_usage():
    if torch.cuda.is_available():
        print(
            f"GPU Usage: {torch.cuda.memory_allocated() / 1e6:.2f} MB allocated / {torch.cuda.memory_reserved() / 1e6:.2f} MB reserved"
        )
        print(f"CUDA Memory Cached: {torch.cuda.memory_cached() / 1e6:.2f} MB")
    else:
        print("CUDA is not available. Running on CPU.")


def consturct_ProPLSTM(base_architecture, num_classes, num_dimensions, use_gcb=True):
    model = ProtoPLSTM(
        num_classes=num_classes, num_dimensions=num_dimensions, use_gcb=use_gcb
    )
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, train_loader: DataLoader, epoch, device, optimizer, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.unsqueeze(1)
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
            data = data.unsqueeze(1)
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


def evaluate(model, loader, device, criterion, scheduler=None):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.unsqueeze(1)
            data, target = data.to(device).float(), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    total_loss /= len(loader.dataset)
    if scheduler:
        scheduler.step(total_loss)
    acc = correct / len(loader.dataset)
    return acc


if __name__ == "__main__":
    print("Starting single run...")

    print("loading datasets")
    X_train, y_train = load_ts_file(train_file)

    from sklearn.model_selection import train_test_split

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    train_dataset = TimeSeriesDataset(X_train_split, y_train_split)
    val_dataset = TimeSeriesDataset(X_val_split, y_val_split)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=False
    )

    X_test, y_test = load_ts_file(test_file)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=False
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = consturct_ProPLSTM(
        base_architecture="convlstm",
        num_classes=num_classes,
        num_dimensions=num_dimensions,
        use_gcb=True,
    )
    model = model.to(device)

    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    r = 5
    best_val_acc = 0
    best_test_acc_at_val_best = 0
    save_name = (
        Path(__file__).resolve().parent.parent.parent
        / f"saved_models/protoplstm/best_models/{dataset_name}/{experiment_run}.pt"
    )
    proto_info_name = (
        Path(__file__).resolve().parent.parent.parent
        / f"saved_models/protoplstm/prototypes/{dataset_name}/{experiment_run}.csv"
    )

    save_name.parent.mkdir(parents=True, exist_ok=True)
    proto_info_name.parent.mkdir(parents=True, exist_ok=True)

    csv_file = (
        Path(__file__).resolve().parent.parent.parent
        / "results/protoplstm/protoplstm_accuracy.csv"
    )

    file_exists = Path(csv_file).exists()

    start_time = time.time()

    for epoch in range(50):
        train(model, train_loader, epoch, device, optimizer, criterion)
        projection = not divmod(epoch, r)[-1] and epoch != 0
        if projection:
            x_all = torch.Tensor(train_dataset[:][0]).unsqueeze(1).to(device)
            y_all = torch.LongTensor(train_dataset[:][1]).to(device)
            model.project_prototypes(x_all, y_all)

        val_acc = evaluate(model, val_loader, device, criterion, scheduler)
        print(f"Validation Accuracy: {val_acc:.4f}")

        test_acc = evaluate(model, test_loader, device, criterion)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc_at_val_best = test_acc
            print(f"New best validation accuracy: {best_val_acc:.4f}")
            torch.save(model.state_dict(), save_name)
            if projection:
                with open(proto_info_name, "w", newline="") as csvfile:
                    protowriter = csv.writer(
                        csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
                    )
                    protowriter.writerow(["prototype", "train_sample", "label"])
                    protowriter.writerows(model.proto_info)

    end_time = time.time()
    elapsed_time = end_time - start_time

    log_data = {
        "dataset_name": dataset_name,
        "best_val_accuracy": best_val_acc,
        "test_accuracy_at_best_val": best_test_acc_at_val_best,
        "training_time_seconds": elapsed_time,
        "training_time_minutes": elapsed_time / 60,
    }

    # Save the training results to the CSV file
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=log_data.keys())

        if not file_exists:  # Write header if file does not exist
            writer.writeheader()

        writer.writerow(log_data)

    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy at Best Validation: {best_test_acc_at_val_best:.4f}")
    print(
        f"Total training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)"
    )
