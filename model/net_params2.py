from torch import nn
from network import MySequential, MyFlattenLayer

net1 = MySequential(
            nn.Conv3d(1, 16, (5, 7, 7), stride=1, padding=0), # in_channels, out_channels, kernel_size
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),  # kernel_size, stride
            nn.Conv3d(16, 32, (5, 7, 7), stride=1, padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            # nn.Conv3d(32, 64, (5, 7, 7), stride=1, padding=0),
            # nn.BatchNorm3d(64),
            # nn.ReLU(),
            # nn.MaxPool3d(2, 2),
            MyFlattenLayer(),
            nn.Linear(32*6*6, 256),
            nn.BatchNorm1d(18, 256),
            nn.ReLU(),
            # nn.Linear(1024, 256),
            # nn.BatchNorm1d(11, 256),
            # nn.ReLU(),
            # nn.Linear(84, 10)
            nn.LSTM(256, 128, num_layers=1, batch_first=True)
            # nn.Linear(512, 64),
            # nn.BatchNorm1d(64),
            # nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU()
        )

net2 = nn.Sequential(
    nn.Linear(5, 1),
    nn.ReLU()
)

net3 = nn.Sequential(
    nn.Linear(129, 1)
    # nn.BatchNorm1d(1, 64),
    # nn.Linear(64, 1),
)