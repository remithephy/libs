'''
:@Author: Remi
:@Date: 2023/8/21 11:18:42
:@LastEditors: Remi
:@LastEditTime: 2023/8/21 11:18:42
:Description: 
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import file_tool as ft
import preprocessing as pp
import BaselineRemoval as BR
 
#####数据读取#####
path = r'D:\20230816\数据汇总\新LTB\ave'
wl,spec = ft.Load_All_Spectral_in_File_Df(path)
z = []
#start = 0
#end = 200
#x,y = x[start:end],y[start:end]
#x,y_uniform = x[start:end],y[start:end]
##################
#anotherway
##################
for i in range(len(spec[0,:])):
    y = spec[:,i]
    #####归一化&平滑#####
    n=10 #拟合阶数    
    y_uniform = pp.Normalization(savgol_filter(y,21,15,mode = "nearest"),method='total_intensity')#SG平滑&归一化
    
    #####baseline#####
    p0 = np.polyfit(wl,y_uniform,n)#多项式拟合，返回多项式系数
    y_fit0 = np.polyval(p0,wl) #计算拟合值
    r0 = y_uniform-y_fit0
    dev0 = np.sqrt(np.sum((r0-np.mean(r0))**2)/len(r0)) #计算残差
    y_remove0 = y_uniform[y_uniform <= y_fit0] #峰值消除
    wl_remove0 = wl[np.where(y_uniform <= y_fit0)] #峰值消除
    i=0
    judge=1
    dev=[]
    while judge:
        p1 = np.polyfit(wl_remove0, y_remove0, n)  # 多项式拟合，返回多项式系数
        y_fit1 = np.polyval(p1, wl_remove0)  # 计算拟合值
        r1 = y_remove0 - y_fit1
        dev1 = np.sqrt(np.sum((r1 - np.mean(r1)) ** 2) / len(r1))  # 计算残差
        dev.append(dev1)
        if i == 0:
            judge = abs(dev[i] - dev0) / dev[i] > 0.05
        else:
            judge = abs((dev[i] - dev[i-1]) / dev[i]) > 0.05 # 残差判断条件
        y_remove0[np.where(y_remove0 <= y_fit1)] = y_fit1[np.where(y_remove0 <= y_fit1)]# 光谱重建
        i=i+1
    y_baseline=np.polyval(p1, wl)  #基线
    y_baseline_correction=y_uniform-y_baseline  #基线校正后
    z.append(y_baseline_correction)

z = np.array(z).T

np.savetxt(path +'\\'+'after_pp.txt',np.hstack((np.array(([wl])).T,z)),delimiter=',',fmt='%.04f')
#######plot######
#plt.figure(1)
#plt.plot(x,y,color="yellow", linewidth=2.0, linestyle="solid", label="Raw_data")
#plt.plot(x,y_filter,color="green", linewidth=2.0, linestyle="solid", label="smooth_data")
#plt.plot(x,y_uniform,color="black", linewidth=2.0, linestyle="solid", label="uni_data")
#plt.plot(x,y_baseline,color="red", linewidth=2.0, linestyle="solid", label="Baseline")
#plt.plot(x,y_baseline_correction,color="blue", linewidth=2.0, linestyle="solid", label="After_correction")
#plt.title('spetrum',fontsize=30)#设置图的标题
#plt.legend(loc="best")#图例放到图中的最佳位置
#plt.xlabel('wavenumber(cm^-1)',fontsize=14)  #设置横轴名称以及字体大小
#plt.ylabel('amplitude',fontsize=14)   #设置纵轴
 
#plt.savefig('a_myplot.jpg', dpi=700) #保存图片，矢量图
#plt.show()