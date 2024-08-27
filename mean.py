'''
:@Author: Remi
:@Date: 2023/8/16 13:29:14
:@LastEditors: Remi
:@LastEditTime: 2023/8/16 13:29:14
:Description: 
'''
import file_tool as ft
import numpy as np


dirpath = r'D:\20230816\数据汇总\新LTB\origin\sliced\6.csv_sliced.csv'
name_list=[]
#路径和名字
wl,spec = ft.Load_MultiSpec(dirpath)
np.savetxt(dirpath + '_ave.csv',np.vstack((wl,np.mean(spec,axis = 1))).T,delimiter=',',fmt='%.06f')