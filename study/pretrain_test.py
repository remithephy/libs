import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from main import data_generate

# 生成预测数据
draw_day = 1825
draw_year = 2020

day_list = [0,366,365,365,365,366]
mounth = [0,31,28,31,30,31,30,31,31,30,31,30,31]

path = r'E:\zjy\shuju\data20240219'
#oco_path = r'F:\zjy\shuju\down\down'
oco_path = r'E:\zjy\shuju\data20240219\down'
tcco2_path = r'E:\zjy\shuju\data20240219\down_cams'
sp_path = r'E:\zjy\shuju\data20240219\down_sp'
d2m_path = r'E:\zjy\shuju\data20240219\down_d2m'
u10_path = r'E:\zjy\shuju\data20240219\down_u10'
v10_path = r'E:\zjy\shuju\data20240219\down_v10'

oco_filelist = os.listdir(oco_path)
oco_year = []
oco_day = []
for filess in oco_filelist:
    years = int('20' + filess[11:13])
    mounths = int(filess[13:15])
    days = sum(day_list[0:years-2015]) + sum(mounth[0:mounths]) + int(filess[15:17])
    if (years%4 == 0 and mounths>2):###闰年情况
        days = days + 1
    oco_year.append(years)
    oco_day.append(days)
# 数据量过大需要分批读取训练
    
split_num = 50 #####分100批
sample_times = len(oco_day)//split_num

# 设置随机种子
random_seed = 2024  
np.random.seed(random_seed)  
random_index = [i for i in range(len(oco_day))]
np.random.shuffle(random_index)

test_day = np.array(oco_day)[random_index[(split_num-1) * sample_times:-1]]   
test_year = np.array(oco_year)[random_index[(split_num-1) * sample_times:-1]]  
test_filelist = np.array(oco_filelist)[random_index[(split_num-1) * sample_times:-1]]


test_data, test_label, test_label_scaler = data_generate(test_filelist,test_day,test_year) # 加了个[0:3]

# tcco2等特征读取
tcco2_path_day = tcco2_path + '\\' + str(draw_day) + '_tcco2.nc'
sp_path_day = sp_path + '\\' + str(draw_day) + '_sp.nc'
d2m_path_day = d2m_path + '\\' + str(draw_day) + '_d2m.nc'
u10_path_day = u10_path + '\\' + str(draw_day) + '_u10.nc'
v10_path_day = v10_path + '\\' + str(draw_day) + '_v10.nc'

tcco2 = xr.open_dataset(tcco2_path_day).tcco2.values.astype(np.float32)
sp = xr.open_dataset(sp_path_day).sp.values.astype(np.float32)
d2m = xr.open_dataset(d2m_path_day).d2m.values.astype(np.float32)
u10 = xr.open_dataset(u10_path_day).u10.values.astype(np.float32)
v10 = xr.open_dataset(v10_path_day).v10.values.astype(np.float32)

for months in day_list:
    if (draw_day-months)>0:
        draw_day = draw_day-months

# 生成1801*3600的经纬度数据
lon = np.around(xr.open_dataset(tcco2_path_day).lon.values.astype(np.float32)- 180,1)#.reshape(1,-1) # 0-360转-180-180
lat = np.around(np.ceil(10 * xr.open_dataset(tcco2_path_day).lat.values.astype(np.float32))/10,1)
new_lon = np.array([lon for _ in range(len(lat))])
new_lat = np.array([lat for _ in range(len(lon))]).T

# 生成day,year 特征 
draw_years = np.array([(draw_year - 2000)/10 for _ in range(len(new_lon))]).reshape(-1,1)
draw_days = np.array([draw_day/360 for _ in range(len(new_lon))]).reshape(-1,1)

# 读取特征数据 
feature = np.hstack((sp, d2m, u10, v10, tcco2, new_lon, new_lat, draw_years, draw_days))
draw_feature = torch.tensor(feature, dtype=torch.float32)

feature_scaler = MinMaxScaler()
features_normalized = feature_scaler.fit_transform(draw_feature)

# 得到预测数据
save_path = path + '\\' + 'model.pth'
loaded_model = torch.load(save_path)

predict_data = loaded_model(features_normalized)
preds_denormalized = test_label_scaler.inverse_transform(predict_data)
df = pd.DataFrame({
    'Predicted': preds_denormalized.flatten(),  
})
df.to_csv('predictions.csv', index=False)

# 绘制地图
data_proj = ccrs.PlateCarree()
# ccrs

xx,yy,zz = lon,lat,preds_denormalized.detach().numpy().reshape(1801,3600)
res = '50m'
fig = plt.figure(4)
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
ax.add_feature(cfeature.OCEAN.with_scale(res), edgecolor='black')
ax.add_feature(cfeature.LAND.with_scale(res), edgecolor='black')
ax.add_feature(cfeature.BORDERS.with_scale(res), edgecolor='black')
ax.coastlines()
ax.contourf(xx,yy,zz,
                transform=data_proj,
                cmap='YlOrRd')
ax.set_global()
#ax.scatter(x, y, marker='o', transform=data_proj)
plt.show()
