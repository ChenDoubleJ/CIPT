import csv
import os
import random

from torch.utils.data import DataLoader, Dataset

from configs import parse_signal_args
import torch
import numpy as np

args = parse_signal_args()
# random seed
fix_seed = args.random_seed
random.seed(fix_seed)


def load_data():
    all_data = []
    bd_data = []  # bd所有的卫星数据
    gal_data = []
    gps_data = []
    n = 4  # 四天
    file_bd = os.listdir("data/BD/")
    file_gps = os.listdir("data/GPS/")
    file_gal = os.listdir("data/GAL/")
    # 读取所有的BD数据
    for i in range(4):
        bd = os.listdir("data/BD/" + file_bd[i])
        length1 = len(bd) - 1  # 每一个文件中的卫星个数文件个数，BD1数据文件中有几个csv存储数据 还有一个是readme文件
        for j in range(length1):
            data = []
            local1 = "data/BD/" + file_bd[i] + "/" + bd[j]  # 每一个卫星的位置
            # # 打开CSV文件进行读取
            with open(local1, 'r', newline='') as file:
                csv_reader = csv.reader(file)
                # 逐行读取数据并存储在data列表中
                for row in csv_reader:
                    if len(row) == 0:
                        continue
                    row = stringTofloat(row)
                    for q in range(len(row)):
                        row[q] = round(row[q], 2)
                    data.append(row)
            if len(data) != 0:
                bd_data.append(data)  # 存放所有的BD卫星的数据
                all_data.append(data)
    print(f"BD数据加载完成！一共{len(bd_data)}个卫星")

    # 读取所有的GAL数据
    for i in range(4):
        gal = os.listdir("data/GAL/" + file_gal[i])
        length2 = len(gal) - 1  # 每一个文件中的卫星个数文件个数，BD1数据文件中有几个csv存储数据 还有一个是readme文件
        for j in range(length2):
            data = []
            local2 = "data/GAL/" + file_gal[i] + "/" + gal[j]
            # # 打开CSV文件进行读取
            with open(local2, 'r', newline='') as file:
                csv_reader = csv.reader(file)
                # 逐行读取数据并存储在data列表中
                for row in csv_reader:
                    row = stringTofloat(row)
                    data.append(row)
            if len(data) != 0:
                gal_data.append(data)
                all_data.append(data)
    print(f"GAL数据加载完成！一共{len(gal_data)}个卫星")

    # 读取GPS数据
    for i in range(4):
        gps = os.listdir("data/GPS/" + file_gps[i])
        # print(len(bd))
        length3 = len(gps) - 1  # 每一个文件中的卫星个数文件个数，BD1数据文件中有几个csv存储数据 还有一个是readme文件
        for j in range(length3):
            data = []
            local3 = "data/GPS/" + file_gps[i] + "/" + gps[j]
            # # 打开CSV文件进行读取
            with open(local3, 'r', newline='') as file:
                csv_reader = csv.reader(file)
                # 逐行读取数据并存储在data列表中
                for row in csv_reader:
                    row = stringTofloat(row)
                    data.append(row)
            if len(data) != 0:
                gps_data.append(data)
                all_data.append(data)
    print(f"GPS数据加载完成！一共{len(gps_data)}个卫星")
    print(f"一共{len(all_data)}个卫星")

    bd_data = random.sample(bd_data, len(bd_data))
    gal_data = random.sample(gal_data, len(gal_data))
    gps_data = random.sample(gps_data, len(gps_data))
    train_data = []
    test_data = []
    l = 0.9
    l1 = 0.9
    train_data.append(bd_data[:int(l * len(bd_data))])
    train_data.append(gal_data[:int(l * len(gal_data))])
    train_data.append(gps_data[:int(l * len(gps_data))])
    test_data.append(bd_data[int(l1 * len(bd_data)):])
    test_data.append(gal_data[int(l1 * len(gal_data)):])
    test_data.append(gps_data[int(l1 * len(gps_data)):])
    tr_data = []
    te_data = []
    for i in range(len(train_data)):
        for l1 in train_data[i]:
            tr_data.append(l1)
    for i in range(len(test_data)):
        for l2 in test_data[i]:
            te_data.append(l2)
    train_data = tr_data
    test_data = te_data
    print(f"训练：{len(train_data)}")
    print(f"测试：{len(test_data)}")
    train = []
    test = []
    L = args.seq_len
    for ls in train_data:
        for j in range(0, len(ls), L):
            if (j + L) > len(ls):
                break
            patch = ls[j:j + L]
            # if len(patch) != L:
            #     break
            train.append(patch)
    print(len(train))
    for ls in test_data:
        for j in range(0, len(ls), L):
            if (j + L) > len(ls):
                break
            patch = ls[j:j + L]
            # if len(patch) != L:
            #     break
            test.append(patch)
    print(len(test))
    train_x, train_y = get_input(train)
    test_x, test_y = get_input(test)

    data_train = DataSet(train_x, train_y)
    data_test = DataSet(test_x, test_y)

    torch.save(data_train, 'data/dataset_train')
    torch.save(data_test, 'data/dataset_test')
    # train_loader = DataLoader(data_train, batch_size=args.batch_size)
    # test_loader = DataLoader(data_test, batch_size=args.batch_size)
    # for i, (x, label) in enumerate(train_loader):
    #     print(x.shape)
    #     print(label.shape)
    # return data_train, data_test


def get_input(X_):
    inputs = []
    y = []
    for x in X_:  # L
        i = []
        v = []
        for ss in x:
            # print(len(ss))
            j = []
            j.append(ss[0])
            j.append(ss[1])
            j.append(ss[2])
            j.append(ss[-2])
            v = [ss[-1]]
            i.append(j)
        y.append(v)
        inputs.append(i)
    return torch.tensor(np.asarray(inputs).astype(np.float32)), torch.LongTensor(np.asarray(y))
    # return inputs, y


class DataSet(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)


def stringTofloat(string_list):
    s = []
    s.append([float(i) for i in string_list])
    s = s[0]
    return s
load_data()
