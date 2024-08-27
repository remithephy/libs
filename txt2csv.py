'''
:@Author: Remi
:@Date: 2023/8/16 13:29:14
:@LastEditors: Remi
:@LastEditTime: 2023/9/12 09:39:55
:Description: 
'''
import file_tool as ft
import preprocessing as pp
import numpy as np
import os
dirpath = r'D:\20230919\origin'
name_list=[]
#路径和名字
samples = ft.Load_Spectral_of_All_Samples_Df(dirpath)
mylist= os.listdir(dirpath)

for name in mylist:
    #这里只是一个相对路径
    name_list.append(dirpath+'\\'+name) 
    name = os.path.join(dirpath,name)

for i in range(len(samples)):   #排除异常值
    print(samples[i][2].size)
    samples[i][2] = pp.Anomalous_spectrum_removal(samples[i][2])
    print(samples[i][2].size)

#for i in range(len(name_list)):
#    np.savetxt(name_list[i] +'\\'+mylist[i]+'.csv',np.hstack((np.array(([samples[i][1]])).T,samples[i][2])),delimiter=',',fmt='%.06f')
#求均值
for i in range(len(name_list)):
   np.savetxt(name_list[i] +'\\'+mylist[i]+'.csv',np.hstack((np.array(([np.mean(samples[i][2],axis=1)])).T,np.hstack((np.array(([samples[i][1]])).T,samples[i][2])))),delimiter=',',fmt='%.06f')    
#只求均值
#for i in range(len(name_list)):
#    np.savetxt(dirpath +'\\'+mylist[i]+'_ave.csv',np.hstack((np.array(([samples[i][1]])).T,(np.array(([np.mean(samples[i][2],axis=1)])).T))),delimiter=',',fmt='%.06f')