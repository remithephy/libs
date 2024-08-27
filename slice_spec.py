'''
:@Author: Remi
:@Date: 2023/8/28 15:42:44
:@LastEditors: Remi
:@LastEditTime: 2023/8/28 15:42:44
:Description: 
'''
import file_tool as ft
import preprocessing as pp
import numpy as np
import pandas as pd
import os
##############path####################
path = r'D:\20230911\para_fit\ave'# only in this case ,use load_multispec
ave_path = r'D:\20230911\para_fit\1_5_0\1_5_0.csv'# maximum the concentration of the element you wanna slice
peak_path = r'D:\20230911\para_fit\Na.csv'
'''
peak_path文件存储方式 eg:

400.871232
410.594123
430.981234

'''
############ init ###########################
name_list = []

slice_list = pd.read_csv(peak_path,header=None,sep = ',').values.flatten(order='c')#read peak list
mylist= os.listdir(path)

############# traverse files in folder #######
for name in mylist:
    if name.endswith(".csv"):
            name_list.append(path+'\\'+name) 
            name = os.path.join(path,name)
############# init line shape ##############
wl,spec = ft.Load_Spectral_Df(ave_path)
p,m,aim_peak = pp.Peak_integrate(wl,spec,slice_list,1)

############## slice&save ####################
for paths in name_list:
    wl,spec = ft.Load_Spectral_Df(paths)
    slice_wl,slice_spec = pp.spec_slice(p,m,aim_peak,wl,spec)
    np.savetxt(paths + '_sliced.csv',np.hstack((slice_wl,np.array(([slice_spec])).T)),delimiter=',',fmt='%.06f')