"""
@date        :    20220525
@author      :    Li Haobo
@Description :    extract adms data, such as No2 data. cal date based on index.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import sys
sys.path.append('/Users/lihaobo/PycharmProjects/ENV')
import lib
import torch
import datetime

"""
folder_path = '/Volumes/8T/AQMS2'
file_list = lib.get_filelist(folder_path, filter=['2020'])
# file_list = file_list[21000:]

count = 0
miss_data = []
for i in file_list:
    # if i[:-16] == '/Volumes/8T/AQMS2/2020/202009/202009':
    #     continue
    try:
        ds = lib.nc_reader(i)
    except Exception as e:
        print(e)
        miss_data.append(i)
        continue
    # ds = lib.nc_reader('/Volumes/8T/AQMS2/2020/202007/20200731/2020073107.nc')
    # print(i)
    try:
        data_no2 = np.vstack((data_no2, np.array(ds.read_value('NO2', 0, 73200))))
    except NameError as e:
        print(e, 'and it will be defined.')
        data_no2 = np.array(ds.read_value('NO2', 0, 73200))
    count += 1
    if count % 1000 == 0:
        print(count)
        # data_name = 'data_no2_part%s.npy' % (count//1000)
        # miss_data_name = 'miss_data_part%s.npy' % (count//1000)
        # np.save(data_name, data_no2)
        # np.save(miss_data_name, miss_data)
        # del data_no2
        # miss_data = []

np.save('data_no2_2020.npy', data_no2)
np.save('miss_data_2020.npy', miss_data)
"""

# data_2019 = np.load('data_no2_2019.npy')
# data_2020 = np.load('data_no2_2020.npy')
# data_2021 = np.load('data_no2_2021.npy')
# miss_2019 = np.load('miss_data_2019.npy')
# miss_2020 = np.load('miss_data_2020.npy')
# miss_2021 = np.load('miss_data_2021.npy')
# a1106 = lib.nc_reader(miss_2019[0])
# a1 = np.array(a1106.read_value('NO2', 0, 73200))
# a0802 = lib.nc_reader(miss_2020[0])
# a2 = np.array(a1106.read_value('NO2', 0, 73200))
# a0721 = lib.nc_reader(miss_2021[0])
# a3 = np.array(a1106.read_value('NO2', 0, 73200))
# a1505 = lib.nc_reader(miss_2021[1])
# a4 = np.array(a1106.read_value('NO2', 0, 73200))
# print(np.max(data_2021))
# print(np.max(data_2020))
# print(np.max(data_2019))
# print(np.min(data_2021))
# print(np.min(data_2020))
# print(np.min(data_2019))
# torch.save(torch.tensor(data_2019), 'data_2019.pt')
# torch.save(torch.tensor(data_2020), 'data_2020.pt')
# torch.save(torch.tensor(data_2021), 'data_2021.pt')

X = torch.load("/Users/lihaobo/PycharmProjects/ENV/NO2/data_2019.pt").float()
X = torch.cat((X, torch.load("/Users/lihaobo/PycharmProjects/ENV/NO2/data_2020.pt").float()))
X = torch.cat((X, torch.load("/Users/lihaobo/PycharmProjects/ENV/NO2/data_2021.pt").float()))
d1 = datetime.datetime(2019, 1, 1, 0)
d2 = datetime.datetime(2019, 8, 11, 6)
interval = d2 - d1
index = interval.days * 24 + interval.seconds // 3600
a = X[index, :]
X = torch.cat((X[:index, :], X[index, :].view(1, -1), X[index:, :]))
d2 = datetime.datetime(2020, 12, 8, 2)
interval = d2 - d1
index = interval.days * 24 + interval.seconds // 3600
X = torch.cat((X[:index, :], X[index, :].view(1, -1), X[index:, :]))
d2 = datetime.datetime(2021, 5, 7, 21)
interval = d2 - d1
index = interval.days * 24 + interval.seconds // 3600
X = torch.cat((X[:index, :], X[index, :].view(1, -1), X[index:, :]))
d2 = datetime.datetime(2020, 8, 15, 5)
interval = d2 - d1
index = interval.days * 24 + interval.seconds // 3600
X = torch.cat((X[:index, :], X[index, :].view(1, -1), X[index:, :]))
torch.save(X, 'data_adms.pt')

# times = np.array([1, 1, 0]).reshape([1, 3])
# for i in range(1, 26304):
#     days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
#     days_leap = days[:]
#     days_leap[1] = 29
#
#     hour_index = i % 24
#     day = i // 24 + 1
#     leap_flag = True
#     if day > 366:
#         day -= 366
#         leap_flag = False
#     if day > 365:
#         day -= 365
#     mon_index = 1
#     day_index = day
#     if day < 32:
#         times = np.append(times, np.array([mon_index, day_index, hour_index]).reshape([1, 3]), axis=0)
#         continue
#     while day > 0:
#         if leap_flag:
#             day -= days_leap[mon_index - 1]
#             mon_index += 1
#         else:
#             day -= days[mon_index - 1]
#             mon_index += 1
#     if leap_flag:
#         mon_index -= 1
#         day_index = day + days_leap[mon_index - 1]
#     else:
#         mon_index -= 1
#         day_index = day + days[mon_index - 1]
#     times = np.append(times, np.array([mon_index, day_index, hour_index]).reshape([1, 3]), axis=0)
#
# torch.save(torch.tensor(times), 'times.pt')



pass

