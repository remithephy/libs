import preprocessing as pp
import os
import pandas as pd
import file_tool as ft
import numpy as np


path = r'D:\20240412\nm\raw'
wl_path = r'D:\20240412\wl.csv'
path_list = ft.find_file(path,'.csv')

filename = [os.path.basename(path_list[i])[:-4] for i in range(len(path_list))]####保留文件名

RSD = []
for paths in path_list:
    raw_feat = pd.read_csv(paths,header=None).values.astype(np.float32)
    raw_wl = pd.read_csv(wl_path,header=None).values.astype(np.float32)
    select_wl = np.where(np.isin(raw_wl,852.075))[0]
    RSD.append(pp.RSD_calculate(raw_feat[select_wl,:].T,6))

np.savetxt(path + '\\' + 'RSD.csv',np.vstack((RSD,filename)),delimiter=',',fmt = '%s')