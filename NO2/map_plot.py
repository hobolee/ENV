"""
@date        :    20220525
@author      :    Li Haobo
@Description :    cal the index of stations in adms. plot adms data. plot wrf data.
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import lib
import numpy as np

# locate stations' location
station_index = []
station_loc = [[22.2819, 114.1822], [22.2833, 114.1557], [22.2868, 114.1429], [22.2845, 114.2169], [22.3586, 114.1271], [22.3147, 114.2233], [22.324, 114.166], [22.4968, 114.1284], [22.378, 114.182], [22.3315, 114.1567], [22.2475, 114.16], [22.4524, 114.162], [22.4728, 114.3583], [22.3177, 114.2594], [22.3733, 114.1121], [22.3908, 113.9767], [22.2903, 113.9411], [22.4467, 114.0203]]
lng_lat = np.load('/Users/lihaobo/PycharmProjects/ENV/lnglat-no-receptors.npz')
lon = lng_lat['lngs'][:73200].reshape([240, 305])
lat = lng_lat['lats'][:73200].reshape([240, 305])
lon = lon[0, :]
lat = lat[:, 0]
for i in range(len(station_loc)):
    tmp = abs(lat - station_loc[i][0])
    tmp_lat = list(np.where(tmp == min(tmp)))[0]
    tmp = abs(lon - station_loc[i][1])
    tmp_lon = list(np.where(tmp == min(tmp)))[0]
    station_index.append([tmp_lat[0], tmp_lon[0]])
print(station_index)

# ADMS 73200
data_2019_01 = np.load('data_no2_2019.npy')[755, :].reshape([240, 305])
lng_lat = np.load('/Users/lihaobo/PycharmProjects/ENV/lnglat-no-receptors.npz')
lon = lng_lat['lngs'][:73200].reshape([240, 305])
lat = lng_lat['lats'][:73200].reshape([240, 305])
fig = plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
cf = plt.contourf(lon, lat, data_2019_01, 60, transform=ccrs.PlateCarree())
ax.coastlines()
# ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
cbar = fig.colorbar(cf, ax=ax, shrink=1)
plt.show()

# wrf
wrf_path = '/Volumes/8T/WRF/data/2021/202101/2021010312/wrfout_d01_2021-01-03_12:00:00'
ds = lib.nc_reader(wrf_path)
lu = ds.ds['U10'][:].squeeze()
lon = ds.ds['XLONG'][:].squeeze()
lat = ds.ds['XLAT'][:].squeeze()
fig = plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
cf = plt.contourf(lon, lat, lu, 60, transform=ccrs.PlateCarree())
ax.coastlines()
# ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
cbar = fig.colorbar(cf, ax=ax, shrink=1)
plt.show()


# ADMS all
# ds = lib.nc_reader('/Volumes/8T/AQMS2/2019/201902/20190201/2019020110.nc')
# data = np.array(ds.read_value('NO2'))[:1200000].reshape([1000, 1200])
# lng_lat = np.load('/Users/lihaobo/PycharmProjects/ENV/lnglat-no-receptors.npz')
# lon = lng_lat['lngs'][:1200000].reshape([1000, 1200])
# lat = lng_lat['lats'][:1200000].reshape([1000, 1200])
# fig = plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())
# cf = plt.contourf(lon, lat, data, 60, transform=ccrs.PlateCarree())
# ax.coastlines()
# # ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
# cbar = fig.colorbar(cf, ax=ax, shrink=1)
# plt.show()
pass
