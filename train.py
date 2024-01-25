import random
import torch
import time

from matplotlib import pyplot as plt
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
# from torch.utils.tensorboard import SummaryWriter
# from dataloader import load_data
from configs import parse_signal_args
import numpy as np
from PatchTST import *
import os

args = parse_signal_args()
# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


def l1_regularization(model, lambda_l1):
    l1_loss = 0.0
    for param in model.parameters():
        l1_loss += torch.norm(param, p=1)
    return lambda_l1 * l1_loss


# L2 正则化函数
def l2_regularization(model, lambda_l2):
    l2_loss = 0.0
    for param in model.parameters():
        l2_loss += torch.norm(param, p=2)
    return lambda_l2 * l2_loss


def load_dataset(datafile):
    data_train = torch.load(datafile + '/dataset_train')
    data_test = torch.load(datafile + '/dataset_test')
    return data_train, data_test


class DataSet(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)


# 测试函数
def test(test_data):
    model.eval()
    loss_allv = 0
    correctv = 0
    batv = 1
    TPv = 0
    FPv = 0
    TNv = 0
    FNv = 0
    with torch.no_grad():
        for i, (xv, labelv) in enumerate(test_data):
            xv = xv.cuda()
            output = model(xv)
            labelv = labelv.cuda()
            l1 = l1_regularization(model, 0.001)
            l2 = l2_regularization(model, 0.001)
            lossv = criterion(output, labelv)
            lossv = lossv + l2
            loss_allv += lossv.item()
            _, predictedv = output.max(1)
            correctv = correctv + ((predictedv.eq(labelv).sum().item()) / (len(labelv)))

            batv += 1
            TPv += ((predictedv == 1) & (labelv.data == 1)).cpu().sum()
            FPv += ((predictedv == 1) & (labelv.data == 0)).cpu().sum()
            TNv += ((predictedv == 0) & (labelv.data == 0)).cpu().sum()
            FNv += ((predictedv == 0) & (labelv.data == 1)).cpu().sum()
        accv = correctv / (batv - 1)
        precicev = TPv / (TPv + FPv)
        recallv = TPv / (TPv + FNv)
        # recall = (TP+TN)/(TP+TN+FP+FN)
        f1scorev = 2 * (precicev * recallv) / (precicev + recallv)
    return loss_allv / (i + 1), accv, precicev, recallv, f1scorev


# 训练函数
def train(train_data):
    model.train()
    loss_all = 0
    correct = 0
    bat = 1
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i, (x, label) in enumerate(train_data):
        optimizer.zero_grad()  # 梯度置零
        x = x.cuda()
        label = label.cuda()
        output = model(x)
        l1 = l1_regularization(model, 0.001)
        l2 = l2_regularization(model, 0.001)
        _, predicted = output.max(1)
        loss = criterion(output, label)
        # loss = mse_loss(predicted.float(),label)
        loss = loss + l2
        loss.backward()
        optimizer.step()
        loss_all += loss.item()

        correct = correct + ((predicted.eq(label).sum().item()) / (len(label)))

        bat += 1

        TP += ((predicted == 1) & (label.data == 1)).cpu().sum()
        FP += ((predicted == 1) & (label.data == 0)).cpu().sum()
        TN += ((predicted == 0) & (label.data == 0)).cpu().sum()
        FN += ((predicted == 0) & (label.data == 1)).cpu().sum()

    acc = correct / (bat - 1)
    precice = TP / (TP + FP)
    recall = TP / (TP + FN)
    # recall = (TP + TN) / (TP + TN + FP + FN)
    f1score = 2 * (precice * recall) / (precice + recall)
    return loss_all / (bat - 1), acc, precice, recall, f1score


# train_data, test_data = load_data()
data_file = "data"
data_train, data_test = load_dataset(data_file)
train_data = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
test_data = DataLoader(data_test, batch_size=args.batch_size, shuffle=False)
print("数据加载完成！")
# writer = SummaryWriter(log_dir='runs‘)  # tensorboard 记录loss变化
criterion = torch.nn.CrossEntropyLoss()  # 定义损失函数
mse_loss = nn.MSELoss()
model = Model(args).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 设置更新adam
start_epoch = 0
# scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # 动态变化
scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-5, last_epoch=start_epoch - 1)
epochs = args.epochs
# epochs = 1
best_loss = 10000
best_acc = 0
losses = []
acces = []
for epoch in range(epochs):
    t1 = time.time()
    loss, acc, precice, recall, f1score = train(train_data)
    scheduler.step()
    t2 = time.time()
    t = t2 - t1
    print(
        '[INFO] Epoch: {}--Time: {:.2f}s--Train: Loss: {:.4f}, Accuracy: {:.4f}, Precice: {:.4f}, recall: {:.4f}, f1sc'
        'ore: {:.4f}'.format(epoch + 1, t, loss, acc, precice, recall, f1score))
    # if epoch > 100 and loss < best_loss and acc > best_acc:
    #     best_loss = loss
    #     best_acc = acc
    # lossv, accv, precicev, recallv, f1scorev = test(test_data)
    torch.save(model.state_dict(), 'best_model/best.pth')
    losses.append(loss)
    acces.append(acc)
    # print("开始测试！")
    model.load_state_dict(torch.load("best_model/best.pth"))
    lossv, accv, precicev, recallv, f1scorev = test(test_data)
    # print("测试结束！")
    if accv > best_acc:
        best_acc = accv
        torch.save(model.state_dict(), 'best_model/best_best.pth')
        print('最好的结果：')
        print('Test: Loss: {:.4f}, Accuracy: {:.4f}, Precice: {:.4f}, recall: {:.4f}, f1sc'
              'ore: {:.4f}'.format(lossv, accv, precicev, recallv, f1scorev))
