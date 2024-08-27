'''
:@Author: Remi
:@Date: 2023/7/23 11:22:38
:@LastEditors: Remi
:@LastEditTime: 2023/8/22 17:11:36
:Description: calculate RSD
'''
import file_tool as ft
import numpy as np
import preprocessing as pp
'''
记得文件夹要用数字命名
'''
################cache/需要改的
WL_ = []
slice_ = []
rsd = []
conc = []

peak_list = [416.595154,413.798187,412.774506,456.283783]##############peak position
tol = 1#算峰面积的容差

##################path##########################
filename = r'D:\20230816\20230816下午oldltb'#file position
##################load#########################
samples=ft.Load_Spectral_of_All_Samples_Df(filename)
#samples[i][j] i = sample_number ,j = 1 -> wavelength,j = 2 -> spectrum,j = 3 -> filename
mean_spec = [np.mean(samples[i][2],axis = 1) for i in range(len(samples))]
p,m,peak_list = pp.Peak_integrate(samples[1][1],mean_spec[0],peak_list,tol)
for i in range(len(samples)):
    for j in range(len(peak_list)):
        WL_.append(np.where(samples[i][1] == peak_list[j])[0][0])#pm wavelength
        slice_.append((samples[i][2][WL_[j] - m[j]:WL_[j] + p[j],:]).mean(axis=0))
        #slice的第0到len(peak_list)行代表sample i 的 j 个peak 的值
        
        length = len(slice_[i*len(peak_list) + j])
        #length = 10
        rsd.append(pp.RSD_calculate(slice_[i*len(peak_list) + j],length))#separate data into (length) parts
    conc.append(samples[i][0])
rsd =np.array(rsd).reshape(len(samples),-1)
conc = [float(x) for x in conc]
np.set_printoptions(precision = 3)
################save################
np.savetxt(filename +'\\'+'RSD.txt',np.hstack((np.array(([conc])).T,rsd)),delimiter=',',fmt='%.04f')