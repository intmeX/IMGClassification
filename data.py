import pandas as pd
import time
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


train_path = './Dataset/Train/'
test_path = './Dataset/Test/'

# 定义数据变换
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def get_data(path, batch_size=20):
    # 按照文件夹获取数据与标签
    data = torchvision.datasets.ImageFolder(
            root=path,
            transform=train_transform
        )

    # 返回DataLoader
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True
    )


def get_train_data(batch_size=10):
    return get_data(path=train_path, batch_size=batch_size)


def get_test_data(batch_size=10):
    return get_data(path=test_path, batch_size=batch_size)

