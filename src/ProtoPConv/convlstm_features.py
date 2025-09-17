from torch import nn
import torch
from src.utils.helpers import load_ts_file
from src.utils.TimeSeriesDataset import TimeSeriesDataset
from src.utils.settings import train_file


class ConvLSTM(nn.Module):
    def __init__(self, filters, filter_size, num_dimensions):
        super().__init__()
        self.conv1 = nn.Conv1d(num_dimensions, filters[0], filter_size, stride=2)
        self.conv2 = nn.Conv1d(filters[0], filters[1], filter_size, stride=2)
        self.conv3 = nn.Conv1d(filters[1], filters[2], filter_size, stride=2)
        self.conv4 = nn.Conv1d(filters[2], filters[3], filter_size, stride=2)

        self.activation = nn.ReLU()
        self.num_dimensions = num_dimensions

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
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
    # inputs = inputs.to(device).float().unsqueeze(1)
    print(f"Input to model: {inputs.shape}")
    outputs = model(inputs)
    print(f"Output from model: {outputs.shape}")
