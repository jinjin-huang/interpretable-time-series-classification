import csv
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Module, ModuleList
from torch.utils.data import DataLoader

from src.utils.helpers import load_ts_file
from src.ProtoPGTN.modules.encoder import Encoder
from src.utils.settings import (
    args,
    dataset_name,
    experiment_run,
    num_classes,
    num_dimensions,
    data_length,
    train_file,
    test_file,
)
from src.utils.TimeSeriesDataset import TimeSeriesDataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Gated_Transformer(Module):
    def __init__(
        self,
        d_model: int,
        d_input: int,
        d_channel: int,
        d_output: int,
        d_hidden: int,
        q: int,
        v: int,
        h: int,
        N: int,
        device: str,
        dropout: float = 0.1,
        pe: bool = False,
        mask: bool = False,
        num_prototypes: int = 10,
    ):
        super(Gated_Transformer, self).__init__()

        self.encoder_list_1 = ModuleList(
            [
                Encoder(
                    d_model=d_model,
                    d_hidden=d_hidden,
                    q=q,
                    v=v,
                    h=h,
                    mask=mask,
                    dropout=dropout,
                    device=device,
                )
                for _ in range(N)
            ]
        )

        self.encoder_list_2 = ModuleList(
            [
                Encoder(
                    d_model=d_model,
                    d_hidden=d_hidden,
                    q=q,
                    v=v,
                    h=h,
                    dropout=dropout,
                    device=device,
                )
                for _ in range(N)
            ]
        )

        self.embedding_channel = torch.nn.Linear(d_channel, d_model)
        self.embedding_input = torch.nn.Linear(d_input, d_model)

        self.gate = torch.nn.Linear(d_model * d_input + d_model * d_channel, 2)
        prototype_dim = d_model * d_input + d_model * d_channel  # matches encoding dim

        # # Modify output linear to take prototype output
        self.output_linear = torch.nn.Linear(prototype_dim, d_output)

        self.pe = pe
        self._d_input = d_input
        self._d_model = d_model

    def forward(self, x, stage):
        # step-wise
        encoding_1 = self.embedding_channel(x)
        input_to_gather = encoding_1

        if self.pe:
            pe = torch.ones_like(encoding_1[0])
            position = torch.arange(0, self._d_input).unsqueeze(-1)
            temp = torch.Tensor(range(0, self._d_model, 2))
            temp = temp * -(math.log(10000) / self._d_model)
            temp = torch.exp(temp).unsqueeze(0)
            temp = torch.matmul(position.float(), temp)
            pe[:, 0::2] = torch.sin(temp)
            pe[:, 1::2] = torch.cos(temp)

            encoding_1 = encoding_1 + pe

        for encoder in self.encoder_list_1:
            encoding_1, score_input = encoder(encoding_1, stage)

        # channel-wise
        encoding_2 = self.embedding_input(x.transpose(-1, -2))
        channel_to_gather = encoding_2

        for encoder in self.encoder_list_2:
            encoding_2, score_channel = encoder(encoding_2, stage)

        encoding_1 = encoding_1.reshape(encoding_1.shape[0], -1)
        encoding_2 = encoding_2.reshape(encoding_2.shape[0], -1)

        # gate
        gate = F.softmax(self.gate(torch.cat([encoding_1, encoding_2], dim=-1)), dim=-1)
        encoding = torch.cat(
            [encoding_1 * gate[:, 0:1], encoding_2 * gate[:, 1:2]], dim=-1
        )

        output = self.output_linear(encoding)

        return (
            output,
            encoding,
            score_input,
            score_channel,
            input_to_gather,
            channel_to_gather,
            gate,
        )


def evaluate(model, loader, stage: str, criterion):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.transpose(1, 2).to(device).float()
            y_batch = y_batch.to(device).long()
            logits, *_ = model(x_batch, stage=stage)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    return (correct / max(1, total)), (total_loss / max(1, len(loader)))


if __name__ == "__main__":
    set_seed(args.random_seed)

    X_train, y_train = load_ts_file(train_file)
    X_test, y_test = load_ts_file(test_file)

    val_ratio = 0.2
    y_np = np.array(y_train)
    train_idx, val_idx = [], []
    for c in np.unique(y_np):
        idx_c = np.where(y_np == c)[0]
        np.random.shuffle(idx_c)
        n_val = max(1, int(len(idx_c) * val_ratio))
        val_idx.extend(idx_c[:n_val].tolist())
        train_idx.extend(idx_c[n_val:].tolist())

    full_train_ds = TimeSeriesDataset(X_train, y_train)
    train_dataset = torch.utils.data.Subset(full_train_ds, train_idx)
    val_dataset = torch.utils.data.Subset(full_train_ds, val_idx)
    test_dataset = TimeSeriesDataset(X_test, y_test)

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
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.random_seed)

    model = Gated_Transformer(
        d_model=args.d_model,
        d_input=data_length,
        d_channel=num_dimensions,
        d_output=num_classes,
        d_hidden=args.d_hidden,
        q=args.q,
        v=args.v,
        h=args.h,
        N=args.N,
        device=device,
        dropout=args.dropout,
        pe=bool(args.pe),
        mask=bool(args.mask),
        num_prototypes=args.num_prototypes,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    num_epochs = args.num_epochs

    best_val_acc = 0.0
    best_val_epoch = -1
    test_acc_at_best_val = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.transpose(1, 2).to(device).float()
            y_batch = y_batch.to(device).long()

            optimizer.zero_grad()
            logits, *_ = model(x_batch, stage="train")
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        val_acc, val_loss = evaluate(
            model, val_loader, stage="val", criterion=criterion
        )
        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            test_acc_at_best_val, _ = evaluate(
                model, test_loader, stage="test", criterion=criterion
            )
            print(
                f"New best Val Acc at epoch {best_val_epoch}. "
                f"Test Acc now: {test_acc_at_best_val*100:.2f}% (model saved)"
            )

    final_test_acc, _ = evaluate(model, test_loader, stage="test", criterion=criterion)
    print(
        f"\n Best Val Acc: {best_val_acc*100:.2f}% at epoch {best_val_epoch}\n"
        f" Test Acc at Best Val: {test_acc_at_best_val*100:.2f}%\n"
        f" Final Test Acc (last epoch): {final_test_acc*100:.2f}%"
    )

    results_file = "results/Ablation_test/GTN_accuracy.csv"
    write_header = not os.path.exists(results_file)

    with open(results_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["dataset", "test_acc_at_best_val"])
        writer.writerow([dataset_name, f"{test_acc_at_best_val*100:.2f}"])
    print(f"Results saved to {results_file}")
