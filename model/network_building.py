"""
@date        :    20220525
@author      :    Li Haobo
@Description :    old code to train model.
"""

import torch
from torch import nn, optim
from collections import OrderedDict
# import d2lzh_pytorch as d2l
import random
import time
import numpy as np


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
            nn.MaxPool3d(2, 2), # kernel_size, stride
            # nn.Conv3d(16, 32, (5, 7, 7), stride=1, padding=0),
            # nn.BatchNorm3d(32),
            # nn.ReLU(),
            # nn.MaxPool3d(2, 2),
            # nn.Conv3d(32, 64, (5, 7, 7), stride=1, padding=0),
            # nn.BatchNorm3d(64),
            # nn.ReLU(),
            # nn.MaxPool3d(2, 2),
            MyFlattenLayer(),
            nn.Linear(16*17*17, 2048),
            nn.BatchNorm1d(47, 2048),
            nn.ReLU(),
            # nn.Linear(1024, 256),
            # nn.BatchNorm1d(11, 256),
            # nn.ReLU(),
            #nn.Linear(84, 10)
            nn.LSTM(2048, 512, num_layers=1, batch_first=True),
            # nn.Linear(512, 64),
            # nn.BatchNorm1d(64),
            # nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

net2 = nn.Sequential(
    nn.Linear(4, 1)
)

net3 = nn.Sequential(
    nn.Linear(257, 64),
    # nn.BatchNorm1d(1, 64),
    nn.Linear(64, 1)
)


def data_iter_random(X, x_daily, batch_size, num_steps, device=None):
    num_examples = (len(X) - 720 - 24)
    #     print('examples', num_examples)
    epoch_size = num_examples // batch_size
    #     print('epoch', epoch_size)
    example_indices = list(range(num_examples))

    #     print(example_indices)
    # random.shuffle(example_indices)

    def _data(pos, data, loc):
        pos += 720
        lon = int((loc//5) % 305)
        lat = int((loc//5) // 305 * 5)
        if data == 'x':
            #             print(pos, pos+num_steps)
            #             print(data[pos:pos + num_steps, :, :, :].size())

            tmp = x_daily[pos//24-30:pos//24-3, :, lat-20:lat+20, lon-20:lon+20]
            return torch.cat((tmp, X[pos - 72:pos, :, lat-20:lat+20, lon-20:lon+20]), 0)
        if data == 'y':
            #             print(pos)
            return X[pos + 24, :, lat, lon]

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(epoch_size):
        for j in range(0, 73200, 25):
            lon = int((j // 5) % 305)
            lat = int((j // 5) // 305 * 5)
            if lon < 20 or lon > 305-20:
                continue
            if lat < 20 or lat > 240-20:
                continue
            print(i, j, lon, lat)

            # 每次读取batch_size个随机样本
            ii = i * batch_size
            batch_indices = example_indices[ii: ii + batch_size]
            #         print(batch_indices)
            XX = [_data(index, 'x', j) for index in batch_indices]
            YY = [_data(index, 'y', j) for index in batch_indices]
            XX2 = torch.tensor([[lat, lon, (i+tmp)//30+1, (i+tmp) % 24] for tmp in range(batch_size)])
            XX = torch.stack(XX)
            YY = torch.stack(YY)
            XX = XX.transpose(1, 2)
            yield XX, YY, XX2.float()


def train_ch5(net ,net2, net3, batch_size, optimizer, device, num_epochs):
    global X
    net = net.to(device)
    net2 = net2.to(device)
    net3 = net3.to(device)
    print("training on ", device)
    loss = torch.nn.MSELoss(reduction='mean')
    batch_count = 0
    L = []
    for epoch in range(num_epochs):
        train_l_sum, start = 0.0, time.time()
        train_iter = data_iter_random(X[:, :, :, :], x_daily, batch_size, 99)
        for x, y, x2 in train_iter:
            x = x.to(device)
            x2 = x2.to(device)
            # X = X.to(device)
            y = y.to(device).view(-1)
            y1 = net(x)
            y2 = net2(x2)
            y_hat = net3(torch.cat((y1, y2.view(-1, 1)), 1))
            y_hat = y_hat.view(y_hat.shape[0])
            # print(y.size())
            # print(y_hat.size())

            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            batch_count += 1
        # print(y_hat, y, l)
        L.append(train_l_sum / batch_count / batch_size)
        print('epoch %d, loss %.4f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, time.time() - start))
        # x_test = X[1200:1230, :, :, :]
        # x_test = x_test.transpose(0, 1)
        # x_test = x_test.view(1, 3, 30, 57, 57)
        # print(net(x_test))
        # x_test = X[0:30, :, :, :]
        # x_test = x_test.transpose(0, 1)
        # x_test = x_test.view(1, 3, 30, 57, 57)
        # print(net(x_test))
        torch.save({'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': L},
                   'model.pt')

global X, x_daily
X = torch.load("/Users/lihaobo/PycharmProjects/ENV/NO2/data_2019.pt").float()[:, :]
# Y = torch.load("/Volumes/OS/stock/Y.pt").float()
# Y = X[:, 0]
X = X.view(-1, 1, 240, 305)
for i in range(len(X) // 24 - 1):
    if i == 0:
        x_daily = X[i * 24:(i + 1) * 24, :, :, :].mean(0, True)
    else:
        tmp = X[i * 24:(i + 1) * 24, :, :, :].mean(0, True)
        x_daily = torch.cat((x_daily, tmp), 0)

mode = 0
if mode:
    lr, num_epochs = 0.00001, 500
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    batch_size = 1
    device = 'cpu'
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    checkpoint = torch.load('/Volumes/OS/stock/model2.pt')
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    l = checkpoint['loss']
    e = checkpoint['epoch']
    net.train()
else:
    lr, num_epochs = 0.0001, 500
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    batch_size = 32
    device = 'cpu'

train_ch5(net, net2, net3, batch_size, optimizer, device, num_epochs)
