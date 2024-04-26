import torch.nn as nn


class CNNSurfaceClassifier(nn.Module):
    def __init__(self, input_size=None, output_size=None, kernel_size=3):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size

        self.conv_pre = nn.Sequential(
            nn.Conv1d(in_channels=self.input_size, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.blocks = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.fc_post = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            # nn.Linear(in_features=512, out_features=self.output_size),
            # nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_pre(x)
        x = self.blocks(x)
        return self.fc_post(x)
