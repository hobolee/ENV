"""
@date        :    20220525
@author      :    Li Haobo
@Description :    Build network
"""
import torch
from torch import nn
from collections import OrderedDict
import random
import torch.utils.data as data
import os


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


def load_adms(root):
    # Load MNIST dataset for generating training data.
    path = os.path.join(root, 'data_adms.pt')
    date_path = os.path.join(root, 'times.pt')
    adms = torch.load(path).float()[:, :].view(-1, 1, 240, 305)
    date = torch.load(date_path).float()[:, :]
    return adms, date


class ADMS(data.Dataset):
    def __init__(self, root, is_train, mode):
        """

        :param root: data folder path
        :param is_train:
        :param mode: train, valid, test, all. data contain unchanged and change the index in different mode.
        """
        super(ADMS, self).__init__()

        # 18 stations' index [lat, lon]
        # self.station_index = [[78, 182], [79, 168], [81, 162], [80, 199], [120, 154], [96, 202], [101, 173], [195, 154],
        #                  [130, 181], [105, 169], [59, 170], [171, 171], [182, 270], [98, 221], [128, 146], [137, 78],
        #                  [83, 60], [168, 100]]
        self.station_index = [[78, 182]]
        self.station_num = 1

        self.adms, self.date = load_adms(root)
        self.length = len(self.adms) - 168 - 24
        self.example_indices = list(range(self.length))

        # keep the same shuffle result, train:valid:test = 8:1:1
        r = random.random
        random.seed(2)
        if mode != 'all':
            random.shuffle(self.example_indices, random=r)
        print(self.example_indices[:20])
        self.mode = mode
        if self.mode == 'train':
            self.length = 8 * self.length // 10
            self.example_indices = self.example_indices[:self.length]
        elif self.mode == 'valid':
            self.length = self.length // 10
            self.example_indices = self.example_indices[8*self.length:9*self.length]
        elif self.mode == 'test':
            self.length = self.length - 9 * self.length // 10
            self.example_indices = self.example_indices[-self.length:]
        self.length = self.length * self.station_num
        self.is_train = is_train

        # calculate the daily data
        for i in range(len(self.adms) // 24 - 1):
            if i == 0:
                self.x_daily = self.adms[i * 24:(i + 1) * 24, :, :, :].mean(0, True)
            else:
                tmp = self.adms[i * 24:(i + 1) * 24, :, :, :].mean(0, True)
                self.x_daily = torch.cat((self.x_daily, tmp), 0)

    def __getitem__(self, idx):
        # t_index: index of hour from 2019010100. s_index: index of location
        t_index = idx // self.station_num
        s_index = self.station_index[idx % self.station_num]
        tmp1 = self.x_daily[t_index//4//24:t_index//4//24+4, :, s_index[0]-20:s_index[0]+20, s_index[1]-20:s_index[1]+20]
        tmp2 = self.adms[t_index//4+96:t_index//4+168, :, s_index[0]-20:s_index[0]+20, s_index[1]-20:s_index[1]+20]
        input1 = torch.cat((tmp1, tmp2)).transpose(0, 1)
        target = self.adms[t_index//4+168+24, :, s_index[0], s_index[1]]
        tmp1 = self.date[t_index//4+168+24, :].view(1, 3)
        tmp2 = torch.tensor(s_index[0]).view([1, 1])
        tmp3 = torch.tensor(s_index[1]).view([1, 1])
        input2 = torch.cat((tmp1, tmp2, tmp3), axis=1)
        return idx, target, input1, input2,

    def __len__(self):
        # it will multipy self.station_num
        return self.length


class Model(nn.Module):
    def __init__(self, net1, net2, net3):
        super(Model, self).__init__()
        self.net1 = net1
        self.net2 = net2
        self.net3 = net3

    def forward(self, input1, input2):
        input1 = self.net1(input1)
        input2 = self.net2(input2)
        output = self.net3(torch.cat((input1, input2.view(-1, 1)), 1))
        return output
