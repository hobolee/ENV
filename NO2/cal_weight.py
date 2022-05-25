"""
@date        :    20220525
@author      :    Li Haobo
@Description :    Cal IDW wight using matrix method.
"""

import numpy as np
import pandas as pd
from lib import DataInterpolate


def cal_lon_lat(i, j):
    return 22.1 + i * 0.001, 113.8 + j * 0.001

dt = pd.read_csv("AQ_NO2-19800101-20220504.csv", header=4, dtype=str)
dt = dt.values
index = 0
DI = DataInterpolate(dt, index)
# DI.origin_data = np.delete(DI.origin_data, [7, 10, 13, 15], axis=1)  # delete data in raw file
DI.generate_dataset(48, [2000, 2021])

weight = np.zeros([600, 800, 14])
d = np.zeros([600, 800, 14])
for i in range(600):
    print(i)
    for j in range(800):
        for k in range(14):
            d[i, j, k] = np.linalg.norm(np.array(cal_lon_lat(i, j)) - DI.location[:, k].astype(float))
        for k in range(14):
            weight[i, j, k] = (1 / d[i, j, k]) ** 2 / np.sum((1 / d[i, j, :]) ** 2)

np.save('weight.npy', weight)
pass
