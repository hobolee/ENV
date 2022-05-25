"""
@date        :    20220525
@author      :    Li Haobo
@Description :    Eval the model
"""

from network import ADMS, Model
import torch
import argparse
from net_params import net1, net2, net3
from tqdm import tqdm
import os
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from lib import DataInterpolate
import pandas as pd

def eval():
    '''
    eval the model with test dataset
    :return: save the pred_list, label_list, train_loss, valid_loss
    '''
    TIMESTAMP = "2022-05-20T00-00-00"
    save_dir = './save_model/' + TIMESTAMP
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',
                        default=16,
                        type=int,
                        help='mini-batch size')
    args = parser.parse_args()

    root = '/Users/lihaobo/PycharmProjects/ENV/NO2/'
    testFolder = ADMS(is_train=False, root=root, mode='test')
    testLoader = torch.utils.data.DataLoader(testFolder,
                                             batch_size=args.batch_size,
                                             shuffle=False)

    net = Model(net1, net2, net3)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    print('==> loading existing model')
    model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
    net.load_state_dict(model_info['state_dict'])
    optimizer = torch.optim.Adam(net.parameters())
    optimizer.load_state_dict(model_info['optimizer'])
    train_loss = model_info['train_loss']
    valid_loss = model_info['valid_loss']
    lossfunction = nn.MSELoss()

    # to track the validation loss as the model trains
    test_losses = []
    label_list, pred_list = [], []

    with torch.no_grad():
        net.eval()
        t = tqdm(testLoader, leave=False, total=len(testLoader))
        for i, (idx, targetVar, inputVar1, inputVar2) in enumerate(t):
            inputs1 = inputVar1.to(device)
            inputs2 = inputVar2.to(device)
            label = targetVar.to(device).view(-1)
            pred = net(inputs1, inputs2).squeeze()
            label_list += list(label)
            pred_list += list(pred)
            loss = lossfunction(pred, label)
            loss_aver = loss.item() / args.batch_size
            test_losses.append(loss_aver)
            t.set_postfix({
                'testloss': '{:.6f}'.format(loss_aver)
            })
        test_loss = np.average(test_losses)
        print_msg = f'test_loss: {test_loss:.6f} '
        print(print_msg)

    res = [pred_list, label_list, train_loss, valid_loss]
    np.save('eval_result', res)


def eval_all():
    '''
        eval the model with all dataset
        :return: save the pred_list, label_list, train_loss, valid_loss
        '''
    TIMESTAMP = "2022-05-20T00-00-00"
    save_dir = './save_model/' + TIMESTAMP
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',
                        default=16,
                        type=int,
                        help='mini-batch size')
    args = parser.parse_args()

    root = '/Users/lihaobo/PycharmProjects/ENV/NO2/'
    allFolder = ADMS(is_train=False, root=root, mode='all')
    allLoader = torch.utils.data.DataLoader(allFolder,
                                            batch_size=args.batch_size,
                                            shuffle=False)

    net = Model(net1, net2, net3)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    print('==> loading existing model')
    model_info = torch.load(os.path.join(save_dir, 'checkpoint.pth.tar'))
    net.load_state_dict(model_info['state_dict'])
    optimizer = torch.optim.Adam(net.parameters())
    optimizer.load_state_dict(model_info['optimizer'])
    train_loss = model_info['train_loss']
    valid_loss = model_info['valid_loss']
    lossfunction = nn.MSELoss()

    # to track the validation loss as the model trains
    test_losses = []
    label_list, pred_list = [], []

    with torch.no_grad():
        net.eval()
        t = tqdm(allLoader, leave=False, total=len(allLoader))
        for i, (idx, targetVar, inputVar1, inputVar2) in enumerate(t):
            inputs1 = inputVar1.to(device)
            inputs2 = inputVar2.to(device)
            label = targetVar.to(device).view(-1)
            pred = net(inputs1, inputs2).squeeze()
            label_list += list(label)
            pred_list += list(pred)
            loss = lossfunction(pred, label)
            loss_aver = loss.item() / args.batch_size
            test_losses.append(loss_aver)
            t.set_postfix({
                'testloss': '{:.6f}'.format(loss_aver)
            })
        test_loss = np.average(test_losses)
        print_msg = f'test_loss: {test_loss:.6f} '
        print(print_msg)

    res = [pred_list, label_list, train_loss, valid_loss]
    np.save('eval_result', res)


def eval_plot():
    '''
    plot the ture and predict value.
    plot the loss curve
    :return:
    '''
    result = np.load('eval_result.npy', allow_pickle=True)
    plt.figure()
    x = np.arange(len(result[0]))
    plt.plot(x, result[0], x, result[1])
    plt.figure()
    x = np.arange(len(result[2]))
    plt.plot(x, result[2], x, result[3])
    plt.show()


def eval_adms_station():
    '''
    compare the adms data with the station data
    :return:
    '''
    path = '/Users/lihaobo/PycharmProjects/ENV/NO2/data_adms.pt'
    adms = torch.load(path).float()[:, :].view(-1, 240, 305)
    cbr = adms[:, 78, 182]
    cwa = adms[:, 81, 162]
    tpa = adms[:, 171, 171]

    dt = pd.read_csv("../NO2/AQ_NO2-19800101-20220504.csv", header=4, dtype=str)
    dt = dt.values
    index = 9
    DI = DataInterpolate(dt, index)
    DI.generate_dataset(48, [2019, 2021])
    DI.interpolate('interpolate.interp1d', kind='linear')

    plt.figure()
    plt.plot(DI.operate_X, DI.operate_data_linear, DI.operate_X, tpa)
    plt.show()


if __name__ == "__main__":
    # eval()
    # eval_plot()
    # eval_all()
    eval_adms_station()
