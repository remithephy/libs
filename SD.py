'''
:@Author: Remi
:@Date: 2023/7/23 11:22:38
:@LastEditors: Remi
:@LastEditTime: 2023/7/25 14:28:58
:Description: calculate SD
'''
import file_tool as ft
import numpy as np
##################path##########################
filename = r'G:\20230628\49'

##################load#########################
samples=ft.Load_Spectral_of_All_Samples_Df(filename)
#samples[i][j] i = sample_number ,j = 1 -> wavelength,j = 2 -> spectrum,j = 3 -> filename
pm1 = 17#integrate length
pm2 = 30
mean = []
sd = []
conc = []
for i in range(len(samples)):    
    #print(samples[i])
    WL_1 = np.where(samples[i][1] == 500.900024)[0][0]#pm17 wavelength
    WL_2 = np.where(samples[i][1] == 515.311462)[0][0]#pm30

    slice_1 = samples[i][2][WL_1 - pm1:WL_1 + pm1,:]
    slice_2 = samples[i][2][WL_2 - pm2:WL_2 + pm2,:]
    spec_mean = slice_1.mean(axis=0) / slice_2.mean(axis=0)
    mean.append(spec_mean)
    sd.append(ft.SD_calculate(spec_mean,10))#separate data into 10 parts
    conc.append(samples[i][0])
conc = [float(x) for x in conc]

################save################
np.savetxt(filename +'\\'+'SD.txt',np.stack((conc,sd,mean),axis=1),delimiter=',',fmt='%.04f')