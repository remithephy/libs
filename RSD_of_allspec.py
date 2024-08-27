'''
:@Author: Remi
:@Date: 2023/7/23 11:22:38
:@LastEditors: Remi
:@LastEditTime: 2023/9/19 16:42:40
:Description: calculate RSD
'''
import file_tool as ft
import numpy as np
import preprocessing as pp
'''
记得文件夹要用数字命名
'''
'''
example:
filename = r'D:\20230816\20230816下午oldltb

20230816下午oldltb文件夹下有
1
2
3
4
五个子文件夹
子文件夹1中有
001.txt,002.txt,003.txt,004.txt,005.txt.........等txt文件
'''
################cache/需要改的
spec_num = 146
rsd_list = []
head = ['wl']
##################path##########################
filename = r'D:\20230816\20230816下午newltb+ava\LTB\SD\RSD'#file position
##################load#########################
samples=ft.Load_Spectral_of_All_Samples_Df(filename)

for i in range(len(samples)):
    wl = samples[i][1]
    head.append(samples[i][0])
    rsd = [pp.SD_calculate(samples[i][2][wavelength],spec_num) for wavelength in range(len(wl))] 
    if i == 0:
        rsd_list = np.hstack((np.array(([wl])).T,np.array(([rsd])).T))
    else:
        rsd_list = np.hstack((rsd_list,np.array(([rsd])).T))

np.set_printoptions(precision = 4)        

import pandas as pd
df = np.vstack((np.array(([head])),rsd_list))
pd.DataFrame(df).to_csv(filename +'\\'+ 'RSD.csv', index=False, header=False)

