'''
:@Author: Remi
:@Date: 2023/11/15 15:42:40
:@LastEditors: Remi
:@LastEditTime: 2023/12/26 16:46:05
:Description: 
'''
import pandas as pd
import seaborn as sns
import file_tool as ft
import numpy as np
import preprocessing as pp
import matplotlib.pyplot as plt
'''
保存格式：
eg：csv中

1223  1231  123  2323
123    213  123  23123

代表两个sample对应四个峰的峰面积
'''
################cache/需要改的
conc = []

np.set_printoptions(precision = 6)

peak_list = [779.989,852.075]##############peak position
tol = 0#算峰面积的容差


##################path##########################
filename = r'D:\20231218\internal_stand\3N\internal_wei.csv'#file position
savepath = r'D:\20231218\internal_stand\3N'
##################load#########################

wl,spec=ft.Load_MultiSpec(filename)

meanspec = np.mean(spec,axis = 1)############while variable no_grad use mean spec , otherwise chose max spec 
maxspec = np.max(spec,axis = 1)

p,m,peak_list = pp.Peak_integrate(wl,maxspec,peak_list,tol)

##################calculate#####################
for i in range(len(spec[0])):
    for j in range(len(peak_list)):#pm wavelength

        if(p[j] + m[j]) == 1:###################################判断是峰值还是面积
            slice_cache = spec[peak_list[j],i]
            slice_square = 0
        else:
            slice_cache = spec[peak_list[j] - m[j]:peak_list[j] + p[j],i]
            slice_square = (p[j] + m[j]) * (slice_cache[0] + slice_cache[-1])/2

        if j == 0:
            slice_ = slice_cache.sum(axis=0) - slice_square
        else:
            slice_ = np.vstack((slice_,(slice_cache.sum(axis=0) - slice_square)))

        if i == 0:
            conc.append(wl[peak_list[j]])

    if i == 0:
        slice_sample = slice_   
    else: 
        slice_sample = np.hstack((slice_sample,slice_))

###################save#####################################

np.savetxt(savepath +'\\' + 'square.csv',np.hstack((np.array([conc]).T,slice_sample)).T,delimiter=',',fmt='%.04f')  

###################plot#####################################

for j in range(len(peak_list)):
    sns_spec = spec[:][peak_list[j] - m[j]:peak_list[j] + p[j],:]
    sns_wl = wl_cache = wl[peak_list[j] - m[j]:peak_list[j] + p[j]]

##################减阴影
    for i in range(len(sns_spec[0,:])):
        slope = (sns_spec[:,i][0] - sns_spec[:,i][-1])/(sns_wl[0] - sns_wl[-1])
        bias = sns_spec[:,i][0] - slope * sns_wl[0]
        sns_spec[:,i] = sns_spec[:,i] - (sns_wl * slope + bias)

# ##################展区间  画带SD的图
#     for i in range(len(sns_spec[0][:]) - 1):
#         sns_wl = np.column_stack((sns_wl,wl_cache))

#     sns_spec = sns_spec.flatten()
#     sns_wl = sns_wl.flatten()
#     data = np.column_stack((sns_wl, sns_spec))
#     df = pd.DataFrame(data, columns=['wl', 'int'])
#     sns.relplot(x="wl", y="int", kind="line", data=df,ci = "sd")
#     sns.relplot(x="wl", y="int", kind="line", data=df)
#     plt.show()
##################### 画不同浓度梯度的图
    with plt.style.context('ggplot'):
        linespace = len(sns_spec[0,:])
        for colors in range(linespace):
            plt.plot(sns_wl,sns_spec[:,colors],color= plt.get_cmap('Blues')((np.linspace(0, 1, linespace)))[colors])
        plt.xlabel("Wl")
        plt.ylabel("I")
        plt.show()