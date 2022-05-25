"""
@date        :    20220525
@author      :    Li Haobo
@Description :    Cal IDW wight and plot spatial interpolation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import sys
# sys.path.append('/Users/lihaobo/PycharmProjects/ENV_AQ')
from lib import DataInterpolate
import cartopy.crs as ccrs


# calculate lon and lat according to index
def cal_lon_lat(i, j):
    return 22.1 + i * 0.001, 113.8 + j * 0.001


dt = pd.read_csv("AQ_NO2-19800101-20220504.csv", header=4, dtype=str)
dt = dt.values
index = 0
DI = DataInterpolate(dt, index)
# DI.origin_data = np.delete(DI.origin_data, [7, 10, 13, 15], axis=1)  # delete data in raw file
DI.generate_dataset(48, [2010, 2021])
print(DI.origin_data.shape)
print(DI.operate_data.shape)

# calculate IDW's weight
DI.operate_data = DI.operate_data[-100:, :]
s_distribution = np.zeros((600, 800, DI.operate_data.shape[0]))
# for i in range(s_distribution.shape[0]):
#     print(i)
#     for j in range(s_distribution.shape[1]):
#         lat_lon = np.array(cal_lon_lat(i, j))
#         dis = [np.linalg.norm(lat_lon - DI.location[:, index].astype(float)) for index in range(14)]
#         # for k in range(s_distribution.shape[2]):
#         for k in range(12, 13):
#             if sum(DI.operate_data[k, :]) > 0:
#                 for num in range(len(DI.operate_data[1])):
#                     d_weight = 1 / dis[num] / (sum([1/d for d in dis]))
#                     s_distribution[i, j, k] += (DI.operate_data[k, num]) * d_weight

# plot spatial interpolation with IDW weight based on 18 stations
weight = np.load('weight.npy')
weight = weight.reshape([-1, 14])
# for k in range(len(DI.operate_data)):
#     if k % 999 == 0:
#         print(k)
#     if sum(DI.operate_data[k, :]) > 0:
#         tmp = np.dot(weight, DI.operate_data.transpose())
#         # tmp = np.sum(tmp, axis=1)
#         s_distribution[:, :, k] = tmp.reshape([600, 800, -1])

tmp = np.dot(weight, DI.operate_data.transpose())
# tmp = np.sum(tmp, axis=1)
s_distribution = tmp.reshape([600, 800, -1])
ylist = np.linspace(22.1, 22.699, 600)
xlist = np.linspace(113.8, 114.599, 800)
X, Y = np.meshgrid(xlist, ylist)
fig = plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
cp = plt.contourf(X, Y, s_distribution[:, :, -1], 60, transform=ccrs.PlateCarree())
ax.coastlines()
cbar = fig.colorbar(cp, ax=ax, shrink=1)# ax.set_title('Filled Contours Plot')
cbar.ax.set_ylabel(r'NO2(ppb)')
ax.set_xlabel('lon')
ax.set_ylabel('lat')
plt.show()
# location = np.zeros([600, 800, 2])
# for i in range(600):
#     for j in range(800):
#         location[i, j, :] = cal_lon_lat(i, j)

# np.savetxt('no22.csv', np.stack((s_distribution[::10, ::10, 12], X[::10, ::10], Y[::10, ::10]), axis=2).reshape([4800, 3]))
pass
