from torch import nn
import torch
from src.utils.helpers import load_ts_file
from src.utils.TimeSeriesDataset import TimeSeriesDataset
from src.utils.settings import train_file


class ConvLSTM(nn.Module):
    def __init__(self, filters, filter_size, num_dimensions):
        super().__init__()
        self.conv1 = nn.Conv2d(1, filters[0], (1, filter_size), stride=(1, 2))
        self.conv2 = nn.Conv2d(filters[0], filters[1], (1, filter_size), stride=(1, 2))
        self.conv3 = nn.Conv2d(filters[1], filters[2], (1, filter_size), stride=(1, 2))
        self.conv4 = nn.Conv2d(filters[2], filters[3], (1, filter_size), stride=(1, 2))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.num_dimensions = num_dimensions

        self.rnn = nn.GRU(
            filters[3] * self.num_dimensions,
            int(filters[3] / 4 * self.num_dimensions),
            2,
            bidirectional=True,
            dropout=0.4,
        )

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        x = self.activation(self.conv1(x))
        # print(f"After conv1: {x.shape}")
        x = self.activation(self.conv2(x))
        # print(f"After conv2: {x.shape}")
        x = self.activation(self.conv3(x))
        # print(f"After conv3: {x.shape}")
        x = self.activation(self.conv4(x))
        # print(f"After conv4: {x.shape}")

        x = x.view(x.size(0), -1, x.size(-1))
        x = x.permute(2, 0, 1)
        x, _ = self.rnn(x)
        # print(f"After RNN: {x.shape}")

        x = x.permute(1, 2, 0)
        x = x.view(x.size(0), -1, self.num_dimensions, x.size(-1))
        # print(f"After reshape: {x.shape}")

        return x


def convlstm_features(**kwargs):
    """Constructs a convlstem model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return ConvLSTM([32, 64, 128, 256], 5, **kwargs)


if __name__ == "__main__":

    X_train, y_train = load_ts_file(train_file)
    train_dataset = TimeSeriesDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=False
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = convlstm_features(num_dimensions=9)
    model = model.to(device)

    sample_batch = next(iter(train_loader))
    inputs, labels = sample_batch
    inputs = inputs.to(device).float()
    inputs = inputs.to(device).float().unsqueeze(1)
    print(f"Input to model: {inputs.shape}")
    outputs = model(inputs)
    print(f"Output from model: {outputs.shape}")
