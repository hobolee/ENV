"""
@date        :    20220525
@author      :    Li Haobo
@Description :    old code to train model.
"""

import platform
print(platform.platform())
import sys
print(sys.version)
import torch
print(torch.__version__)
from torch import nn, optim
from collections import OrderedDict
# import d2lzh_pytorch as d2l
import random
import time
import numpy as np
import datetime


class MyFlattenLayer(torch.nn.Module):
    def __init__(self):
        super(MyFlattenLayer, self).__init__()

    def forward(self, x): # x shape: (batch, *, *, ...)
        x = x.transpose(1, 2)
        return x.reshape(x.shape[0], x.shape[1], -1)


class MySequential(nn.Module):
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict): # 如果传入的是一个OrderedDict
            for key, module in args[0].items():
                self.add_module(key, module)  # add_module方法会将module添加进self._modules(一个OrderedDict)
        else:  # 传入的是一些Module
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input):
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成
        for module in self._modules.values():
            if type(module) is torch.nn.modules.rnn.LSTM:
                # input = input.view(-1, 30, 84*4)
                input, (h_n, c_n) = module(input)
                input = input[:, -1, :]
                # print('lstm', input.size())
            else:
                input = module(input)
                # print('other', input.size())
        return input


net = MySequential(
            nn.Conv3d(1, 16, (5, 7, 7), stride=1, padding=0), # in_channels, out_channels, kernel_size
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),  # kernel_size, stride
            # nn.Conv3d(16, 32, (5, 7, 7), stride=1, padding=0),
            # nn.BatchNorm3d(32),
            # nn.ReLU(),
            # nn.MaxPool3d(2, 2),
            # nn.Conv3d(32, 64, (5, 7, 7), stride=1, padding=0),
            # nn.BatchNorm3d(64),
            # nn.ReLU(),
            # nn.MaxPool3d(2, 2),
            MyFlattenLayer(),
            nn.Linear(16*7*7, 512),
            nn.BatchNorm1d(47, 512),
            nn.ReLU(),
            # nn.Linear(1024, 256),
            # nn.BatchNorm1d(11, 256),
            # nn.ReLU(),
            # nn.Linear(84, 10)
            nn.LSTM(512, 256, num_layers=1, batch_first=True),
            # nn.Linear(512, 64),
            # nn.BatchNorm1d(64),
            # nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

net2 = nn.Sequential(
    nn.Linear(5, 1)
)

net3 = nn.Sequential(
    nn.Linear(129, 64),
    # nn.BatchNorm1d(1, 64),
    nn.Linear(64, 1)
)


def data_iter_random(X, x_daily, times, batch_size, num_steps, device=None, mode='train'):
    station_index = [[78, 182], [79, 168], [81, 162], [80, 199], [120, 154], [96, 202], [101, 173], [195, 154],
                     [130, 181], [105, 169], [59, 170], [171, 171], [182, 270], [98, 221], [128, 146], [137, 78],
                     [83, 60], [168, 100]]
    station_ind = [i[0] * 305 + i[1] for i in station_index]
    num_examples = (len(X) - 720 - 24)
    #     print('examples', num_examples)
    #     print('epoch', epoch_size)
    example_indices = list(range(num_examples))

    #     print(example_indices)
    r = random.random
    random.seed(2)
    random.shuffle(example_indices, random=r)
    print(example_indices[:20])
    if mode == 'train':
        epoch_size = (8*num_examples//10) // batch_size
        example_indices = example_indices[:8*len(example_indices)//10]
    elif mode == 'val':
        epoch_size = (num_examples // 10) // batch_size
        example_indices = example_indices[8*len(example_indices)//10:9*len(example_indices)//10]
    elif mode == 'test':
        epoch_size = (num_examples - 9*num_examples//10) // batch_size
        example_indices = example_indices[9*len(example_indices)//10:]

    def _data(pos, data, loc):
        pos += 720
        lon = int((loc//5) % 305)
        lat = int((loc//5) // 305 * 5)
        if data == 'x':
            #             print(pos, pos+num_steps)
            #             print(data[pos:pos + num_steps, :, :, :].size())

            tmp = x_daily[pos//24-30:pos//24-3, :, lat-10:lat+10, lon-10:lon+10]
            return torch.cat((tmp, X[pos - 72:pos, :, lat-10:lat+10, lon-10:lon+10]), 0)
        if data == 'y':
            #             print(pos)
            return X[pos + 24, :, lat, lon]
        if data == 't':
            # a = times[pos, :]
            # b = torch.tensor(lat).view([1, 1])
            return torch.cat((times[pos, :].view(1, 3), torch.tensor(lat).view([1, 1]), torch.tensor(lon).view([1, 1])), axis=1)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # t_s = time.time()
    for i in range(epoch_size):
        # print(i, time.time()-t_s)
        # t_s = time.time()

        # for j in range(0, 73200):
        for j in station_ind:
            # if j not in station_ind:
            #     continue
            lon = int(j % 305)
            lat = int(j // 305)
            # if lon < 20 or lon > 305-20:
            #     continue
            # if lat < 20 or lat > 240-20:
            #     continue

            # 每次读取batch_size个随机样本
            ii = i * batch_size
            batch_indices = example_indices[ii: ii + batch_size]
            #         print(batch_indices)
            XX = [_data(index, 'x', j) for index in batch_indices]
            YY = [_data(index, 'y', j) for index in batch_indices]
            TT = [_data(index, 't', j) for index in batch_indices]
            if i == 13:
                a = 1
            XX = torch.stack(XX)
            YY = torch.stack(YY)
            TT = torch.stack(TT)
            XX = XX.transpose(1, 2)
            yield XX, YY, TT


def train_ch5(net ,net2, net3, batch_size, optimizer, device, num_epochs):
    global X
    net = net.to(device)
    net2 = net2.to(device)
    net3 = net3.to(device)
    print("training on ", device)
    loss = torch.nn.MSELoss(reduction='mean')
    L_train, L_val = [], []
    for epoch in range(num_epochs):
        train_l_sum, val_l_sum, start = 0.0, 0.0, time.time()
        batch_count = 0
        train_iter = data_iter_random(X[:, :, :, :], x_daily, times, batch_size, 99, mode='train')
        for x, y, t in train_iter:
            x = x.to(device)
            t = t.to(device)
            # X = X.to(device)
            optimizer.zero_grad()
            net.train()
            net2.train()
            net3.train()
            y = y.to(device).view(-1)
            y1 = net(x)
            y2 = net2(t)
            y_hat = net3(torch.cat((y1, y2.view(-1, 1)), 1))
            y_hat = y_hat.view(y_hat.shape[0])
            # print(y.size())
            # print(y_hat.size())

            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            batch_count += 1
        loss_train = train_l_sum / batch_count
        L_train.append(loss_train)
        print('epoch %d, train loss %.4f, time %.1f sec'
              % (epoch + 1, loss_train, time.time() - start))

        batch_count, start = 0, time.time()
        with torch.no_grad():
            net.eval()
            val_iter = data_iter_random(X[:, :, :, :], x_daily, times, batch_size, 99, mode='val')
            for x, y, t in val_iter:
                net.eval()
                net2.eval()
                net3.eval()
                x = x.to(device)
                t = t.to(device)
                y = y.to(device).view(-1)
                y1 = net(x)
                y2 = net2(t)
                y_hat = net3(torch.cat((y1, y2.view(-1, 1)), 1))
                y_hat = y_hat.view(y_hat.shape[0])

                l = loss(y_hat, y)
                val_l_sum += l.cpu().item()
                batch_count += 1
            val_train = val_l_sum / batch_count
            L_val.append(val_train)
            print('epoch %d, val loss %.4f, time %.1f sec'
                  % (epoch + 1, val_train, time.time() - start))

        torch.save({'epoch': epoch,
                    'model_state_dict1': net.state_dict(),
                    'model_state_dict2': net2.state_dict(),
                    'model_state_dict3': net3.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': L_train,
                    'val_loss': L_val},
                   'model_all2.pt')


global x_daily, X, times
X = torch.load("/Users/lihaobo/PycharmProjects/ENV/NO2/data_adms.pt").float()[:, :]
X = X.view(-1, 1, 240, 305)
times = torch.load("/Users/lihaobo/PycharmProjects/ENV/NO2/times.pt").float()[:, :]

for i in range(len(X) // 24 - 1):
    if i == 0:
        x_daily = X[i * 24:(i + 1) * 24, :, :, :].mean(0, True)
    else:
        tmp = X[i * 24:(i + 1) * 24, :, :, :].mean(0, True)
        x_daily = torch.cat((x_daily, tmp), 0)

mode = 0
if mode:
    lr, num_epochs = 0.0001, 500
    batch_size = 16
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    checkpoint = torch.load('/Users/lihaobo/PycharmProjects/ENV/model/model_all.pt')
    net.load_state_dict(checkpoint['model_state_dict1'])
    net2.load_state_dict(checkpoint['model_state_dict2'])
    net3.load_state_dict(checkpoint['model_state_dict3'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # l = checkpoint['loss']
    # e = checkpoint['epoch']
    net.train()
else:
    lr, num_epochs = 0.001, 500
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    batch_size = 16
    device = torch.device("cpu")

train_ch5(net, net2, net3, batch_size, optimizer, device, num_epochs)
