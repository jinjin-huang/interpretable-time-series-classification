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
        self.pe = pe
        self._d_input = d_input
        self._d_model = d_model

    def forward(self, x, stage):
        # step-wise
        # score matrix is input, default add mask and pe
        encoding_1 = self.embedding_channel(x)

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
            encoding_1, _ = encoder(encoding_1, stage)

        # channel-wise
        # score matrix is channel-wise, default no mask and pe
        encoding_2 = self.embedding_input(x.transpose(-1, -2))
        for encoder in self.encoder_list_2:
            encoding_2, _ = encoder(encoding_2, stage)
        # 3D to 2D
        encoding_1 = encoding_1.reshape(encoding_1.shape[0], -1)
        encoding_2 = encoding_2.reshape(encoding_2.shape[0], -1)

        # gate
        gate = F.softmax(self.gate(torch.cat([encoding_1, encoding_2], dim=-1)), dim=-1)
        encoding = torch.cat(
            [encoding_1 * gate[:, 0:1], encoding_2 * gate[:, 1:2]], dim=-1
        )

        return encoding
