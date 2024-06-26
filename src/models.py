import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=inplanes, out_channels=outplanes, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm1d(num_features=outplanes)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return out


class Block(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Block, self).__init__()
        self.mapping = nn.Conv1d(in_channels=inplanes, out_channels=outplanes, kernel_size=1, stride=2)
        self.bblock1 = BasicBlock(inplanes=inplanes, outplanes=outplanes, stride=2)
        self.bblock2 = BasicBlock(inplanes=outplanes, outplanes=outplanes, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.mapping(x)

        out = self.bblock1(x)
        out = self.relu(out)
        out = self.bblock2(out)
        out += identity
        out = self.relu(out)
        return out


class CNNSurfaceClassifier(nn.Module):
    def __init__(self, input_size=None, output_size=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.conv_pre = nn.Sequential(
            nn.Conv1d(in_channels=self.input_size, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.block1 = Block(inplanes=32, outplanes=64)
        self.block2 = Block(inplanes=64, outplanes=128)
        self.block3 = Block(inplanes=128, outplanes=256)
        self.fc_post = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(),
        )
        self.classification = nn.Sequential(
            nn.Linear(in_features=256, out_features=self.output_size),
        )

    def forward(self, x):
        x = self.conv_pre(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.fc_post(x)
        x = self.classification(x)
        return x
