import pandas as pd
import time
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import os
import data
import VGG
import traceback


hidden_dropout_prob = 0.3
weight_decay = 0.01
#device的用途：作为Tensor或者Model被分配到的位置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
epoch = 10
batch_size = 10


# 模型加载
# modle_name = "vgg16"
net_model = VGG.vgg(model_name='vgg16', num_classes=10, init_weights=True)
net_model.to(device)


#train_loss, train_acc = training(net_model, train_loader, optimizer, loss_func, device)
def training(model: torch.nn.Module, dataloader, optimizer, criterion, device):
    global tokenizer
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    #enumerate是将一个可遍历的数据对象如列表，转换为一个索引序列，便于for遍历
    #batch是一批，包含batch_size个数据，这个可以调，这一批里面数据的标签是可以不一样的
    for i, batch in enumerate(dataloader):
        img, label = batch
        #print("batch:",batch)
        # img_tensor = torch.tensor(img, dtype=torch.long).to(device)
        img = img.to(device)
        label = label.to(device)

        optimizer.zero_grad() #将这一轮梯度清零，防止这一轮的梯度影响下一轮的更新

        output = model(img)
        prob = output
        pred = prob.argmax(dim=1)

        #计算 loss
        #criterion(prediction,lable) 分类问题中，交叉熵函数是比较常用也是比较基础的损失函数，能够表征真实样本标签和预测概率之间的差值
        loss = criterion(output, label)

        #计算准确度 acc
        acc = ((pred == label.view(-1)).sum()).item()

        loss.backward() #反向计算各参数的梯度
        optimizer.step() #更新网络net_model中的全部参数

        #.item()取出单元素张量的元素值并返回该值，保持原元素类型不变   和直接是epoch_loss += loss相比，主要是精度上的区别
        epoch_loss += loss.item()
        epoch_acc += acc
        print(loss.item(),acc)

        #每10批，输出一次当前的平均loss和准确度acc
        if i % 10 == 9:
            print("{:>5} loss: {}\tacc: {}".format(i, epoch_loss / (i + 1), epoch_acc / ((i + 1) * batch_size)))
    return epoch_loss / len(dataloader), epoch_acc / (len(dataloader) * batch_size)


#eval_loss, eval_acc = evaluting(net_model, validate_loader, loss_func, device)
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

            #optimizer.zero_grad()

            output = model(img)
            prob = output
            pred = prob.argmax(dim=1)

            loss = criterion(output, label)

            acc = ((pred == label.view(-1)).sum()).item()

            #相比上面training函数，主要是没有对网络的参数进行更新，没有去优化这个网络模型

            epoch_loss += loss.item()
            epoch_acc += acc

    return epoch_loss / len(dataloader), epoch_acc / (len(dataloader) * batch_size)


if __name__ == '__main__':
    train_loader = data.get_train_data(batch_size=batch_size)
    validate_loader = data.get_test_data(batch_size=batch_size)
    #print(train_loader.dataset[2][1],train_loader.dataset[300])
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
    #构造优化器  
    #net_model.parameters()会返回网络net_model的全部参数，lr是学习率
    #可以实现只训练模型的一部分参数，以及不同部分的参数设置不同的学习率
    optimizer = AdamW(net_model.parameters(), lr=5e-5)
    loss_func = CrossEntropyLoss()

    times = []
    model_path = './model_vgg/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    last_epoch = 0

    for i in range(epoch):  #迭代epoch次
        torch.cuda.synchronize() #主要就是想要准确的记录时间，在pytorch里面，程序的执行都是异步的，不用这个的话，最后得到的时间会很短
        start = time.time()
        try:
            train_loss, train_acc = training(net_model, train_loader, optimizer, loss_func, device)
            print("\ntraining epoch {:>2} loss: {}\tacc: {}\n".format(i + 1, train_loss, train_acc))
            eval_loss, eval_acc = evaluting(net_model, validate_loader, loss_func, device)
            print("\nevaluting epoch {:<1} loss: {}\tacc: {}\n".format(i + 1, eval_loss, eval_acc))
        except:
            torch.save(net_model.state_dict(), model_path + 'pytorch_model.bin')
            print("except")
            print(traceback.format_exc())
            break

        last_epoch = i + 1
        torch.cuda.synchronize() #dd
        times.append(time.time() - start)

    if last_epoch == epoch:
        torch.save(net_model.state_dict(), model_path + 'pytorch_model.bin')
    print("\ntimecost:", times, "\n")



