import pandas as pd
import time
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import os
import data
import AlexNet


hidden_dropout_prob = 0.3
weight_decay = 0.01
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epoch = 5
batch_size = 10


# 模型加载
net_model = AlexNet.AlexNet(num_classes=10, init_weights=True)
net_model.to(device)


def training(model: torch.nn.Module, dataloader, optimizer, criterion, device):
    global tokenizer
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for i, batch in enumerate(dataloader):
        img, label = batch
        # img_tensor = torch.tensor(img, dtype=torch.long).to(device)
        img = img.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        output = model(img)
        prob = output
        pred = prob.argmax(dim=1)

        # loss = criterion(prob.view(-1, 5), img.view(-1))
        loss = criterion(output, label)

        acc = ((pred == label.view(-1)).sum()).item()

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc

        if i % 10 == 9:
            print("{:>5} loss: {}\tacc: {}".format(i, epoch_loss / (i + 1), epoch_acc / ((i + 1) * batch_size)))
    return epoch_loss / len(dataloader), epoch_acc / (len(dataloader) * batch_size)


def evaluting(model: torch.nn.Module, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            img, label = batch
            # img_tensor = torch.tensor(img, dtype=torch.long).to(device)
            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            output = model(img)
            prob = output
            pred = prob.argmax(dim=1)

            loss = criterion(output, label)

            acc = ((pred == label.view(-1)).sum()).item()

            epoch_loss += loss.item()
            epoch_acc += acc

    return epoch_loss / len(dataloader), epoch_acc / (len(dataloader) * batch_size)


if __name__ == '__main__':
    train_loader = data.get_train_data(batch_size=batch_size)
    validate_loader = data.get_test_data(batch_size=batch_size)
    # print(train_loader.dataset[2])
    # print(validate_loader.dataset[2])
    # optimizer = Adam(lr=1e-4, eps=1e-8, weight_decay=0.01)
    # 参考博客 一是去掉无用的设置 二是构造字典列表以使AdamW可以接受该参数
    """
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    """
    optimizer = AdamW(net_model.parameters(), lr=5e-5)
    loss_func = CrossEntropyLoss()

    times = []
    model_path = './model/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    last_epoch = 0

    for i in range(epoch):
        torch.cuda.synchronize()
        start = time.time()
        try:
            train_loss, train_acc = training(net_model, train_loader, optimizer, loss_func, device)
            print("\ntraining epoch {:>2} loss: {}\tacc: {}\n".format(i + 1, train_loss, train_acc))
            eval_loss, eval_acc = evaluting(net_model, validate_loader, loss_func, device)
            print("\nevaluting epoch {:<1} loss: {}\tacc: {}\n".format(i + 1, eval_loss, eval_acc))
        except:
            torch.save(net_model.state_dict(), model_path + 'pytorch_model.bin')
            break

        last_epoch = i + 1
        torch.cuda.synchronize()
        times.append(time.time() - start)

    if last_epoch == epoch:
        torch.save(net_model.state_dict(), model_path + 'pytorch_model.bin')
    print("\ntimecost:", times, "\n")

