import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=10, init_weights=False):
        super(AlexNet, self).__init__()
        '''
        模型网络分为特征提取器与分类器，事实上这两者的界限不是特别明显。
        在这个AlexNet网络中，分类器与特征提取器的主要区别是全连接层
        '''
        self.features = nn.Sequential(
            # shape: [3, 224, 224]
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            # shape: [48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # shape: [48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            # shape: [128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # shape: [128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            # shape: [192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            # shape: [192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            # shape: [128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # shape: [128, 6, 6]
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(2048, 10)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # 展为一维
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        x = self.linear(x)
        return x

    def transform(self, x):
        x = self.features(x)
        # 展为一维
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class AlexLinear(nn.Module):
    def __init__(self, init_weights=False):
        super(AlexLinear, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(2048, 10),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

