'''
:@Author: Remi
:@Date: 2023/8/21 09:38:16
:@LastEditors: Remi
:@LastEditTime: 2023/8/21 09:38:16
:Description: 
'''
import file_tool as ft
import numpy as np
import preprocessing as pp
import os

path = r'D:\20231017(仿真卤水)\origin'
list = os.listdir(path)
csv_list = []
for name1 in list:
    if np.isin(name1.endswith(".csv"),1):
        csv_list.append(name1)

for name in csv_list:
    WL1,SP1 = ft.Load_MultiSpec(path + '\\' + name)
    #replace nan
    SP1 = SP1[1:].astype(float)
    SP1[np.isnan(SP1)] = 0

    #对除第一列的每列标准化
    for i in range(len(SP1[0][:])):
        sp2 = SP1[:,i:i+1]
        sp2 = pp.Normalization(sp2,method='total_intensity')
        if i == 0:
            All_Spec = sp2    
        else:
            All_Spec = np.hstack((All_Spec,sp2))
    WL1 = (np.array([WL1[1:]]).T).astype(float)
    np.savetxt(path +'\\'+name + 'normalized.csv',np.hstack((WL1,All_Spec)),delimiter=',',fmt='%.06f')
