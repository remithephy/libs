'''
:@Author: Remi
:@Date: 2023/10/11 19:30:48
:@LastEditors: Remi
:@LastEditTime: 2023/10/11 19:30:48
:Description: 
'''

import file_tool as ft
import numpy as np
import preprocessing as pp
'''
保存格式：
eg：csv中

1223  1231  123  2323
123    213  123  23123

代表两个sample对应四个峰的峰面积
'''
################cache/需要改的
slice_ = []
conc = []
np.set_printoptions(precision = 6)

peak_list = [416.595154,413.798187,412.774506,456.283783]##############peak position
tol = 2#算峰面积的容差

##################path##########################
filename = r'D:\20230816\20230816下午oldltb'#file position
##################load#########################
samples=ft.Load_Spectral_of_All_Samples_Df(filename)
#samples[i][j] i = sample_number ,j = 1 -> wavelength,j = 2 -> spectrum,j = 3 -> filename
mean_spec = [np.mean(samples[i][2],axis = 1) for i in range(len(samples))]
p,m,peak_list = pp.Peak_integrate(samples[1][1],mean_spec[0],peak_list,tol)


##################calculate#####################
for i in range(len(samples)):
    for j in range(len(peak_list)):#pm wavelength
        if j == 0:
            slice_ = ((samples[i][2][peak_list[j] - m[j]:peak_list[j] + p[j],:]).sum(axis=0))
        else:
            slice_ = np.vstack((slice_,((samples[i][2][peak_list[j] - m[j]:peak_list[j] + p[j],:]).sum(axis=0))))
    
    np.savetxt(filename +'\\'+samples[i][0] + 'calculate.csv',slice_.T,delimiter=',',fmt='%.04f')
    slice_ = []