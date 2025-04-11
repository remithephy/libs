'''
:@author: 12184
:@Date:Created on Thu Apr 20 22:01:34 2023
:@LastEditors: Remi
:@LastEditTime: 2024/1/9 19:09:08
:Description: 
''' 
# -*- coding: utf-8 -*-
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import math
import pywt
import file_tool as ft
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVR

def _calculate_vips(model):####PLS
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))#np.zeros()表示初始化0向量
    s = np.diag(np.matmul(np.matmul(np.matmul(t.T,t),q.T), q)).reshape(h, -1)
    #np.matmul(a,b)表示两个矩阵相乘;np.diag()输出矩阵中对角线上的元素，若矩阵是一维数组则输出一个以一维数组为对角线的矩阵
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
        #np.linarg.norm()表示求范数：矩阵整体元素平方和开根号，不保留矩阵二维特性
        vips[i] = np.sqrt(p*(np.matmul(s.T, weight))/total_s)
        #s.T表示矩阵的转置
    return vips

def optimise_pls_cv(X, Y, n_comp):
    pls = PLSRegression(n_components = n_comp)
    r2 = []
    rmse = []
    mae = []
    for test in range(len(Y)):
        X1 = np.delete(X,test,0)
        Y1 = np.delete(Y,test)
        pls.fit(X1,Y1)
        Ypredict = pls.predict(X).flatten()
        r2.append(1 - ((Y - Ypredict)**2).sum()/((Y - Y.mean())**2).sum())
        rmse.append(np.sqrt(mean_squared_error(Y, Ypredict)))
        mae.append(mean_absolute_error(Y, Ypredict))
    return (Ypredict, r2, rmse, mae)


def plot_metrics(X, Y , ylabel, objective):####已知XY绘图找最小/大值
    with plt.style.context('ggplot'):
        plt.plot(X, np.array(Y), '-v', color='blue', mfc='blue')
        if objective=='min':
            idx = np.argmin(Y)
        else:
            idx = np.argmax(Y)
        plt.plot(X[idx], np.array(Y)[idx], 'P', ms=10, mfc='red')

        plt.xlabel('Pixels')
        plt.xticks = X
        plt.ylabel(ylabel)

    plt.show()
    return X[idx]

###########Lorentzian函数
def lorentzian(xc, x0, A, gamma):##################gamma半高宽，A=峰值，X0 中心波长
    return A * gamma * 2 / ((xc - x0)**2 + gamma**2)

#################双Lorentzian函数
def double_lorentzian(xc, x0, A, gamma, x1, A1, gamma1):
    return lorentzian(xc, x0, A, gamma) + lorentzian(xc, x1, A1, gamma1)


def K_divide(x,y,test_percentage = 0.2):#分层交叉验证 #5折 ： test_percentage = 0.2
    unique_lab = np.unique(y)
    k = int(1//test_percentage+1)
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    ####################第KS折作为验证##################
    for ks in range(k):

    ###################对每个label而言#################
        for iter,labels in enumerate(unique_lab):
            ##################init###############
            indexs = np.where(y == labels)[0]
            split_index = [int(1 + len(indexs) * i * test_percentage // 1) for i in range(k+1)]
            split_index[-1] = split_index[-1] - 2 #确保index不超界
            #######################选择第K折部分#############
            test_index = indexs[split_index[ks]:split_index[ks+1]]
            train_index = np.concatenate((indexs[0:split_index[ks]],indexs[split_index[ks+1]:-1]))

            if iter == 0:
                test_Xs = x[test_index]
                test_Ys = y[test_index]
                train_Xs = x[train_index]
                train_Ys = y[train_index]
            else:
                test_Xs = np.vstack((test_Xs,x[test_index]))
                test_Ys = np.concatenate((test_Ys,y[test_index]))
                train_Xs = np.vstack((train_Xs,x[train_index]))
                train_Ys = np.concatenate((train_Ys,y[train_index]))
                
        train_X.append(train_Xs)
        train_Y.append(train_Ys)
        test_X.append(test_Xs)
        test_Y.append(test_Ys)
    return train_X,train_Y,test_X,test_Y

def data_shuffer(X,Y):
    shuffer_index = random.sample(range(0,len(Y)),len(Y))
    shufd_X = X[shuffer_index]
    shufd_Y = Y[shuffer_index]
    return shufd_X,shufd_Y

def optimise_pls_cv(X, Y, n_comp):
    # Define PLS object
    pls = PLSRegression(n_components = n_comp)

    test = 4
    X1 = np.delete(X,test,0)
    Y1 = np.delete(Y,test)
    pls.fit(X1,Y1)

    #pls.fit(X,Y)
    Ypredict = pls.predict(X).flatten()
    #y_cv = cross_val_predict(pls, X, Y, cv=6)
    
    # Calculate scores
    r2 = 1 - ((Y - Ypredict)**2).sum()/((Y - Y.mean())**2).sum()
    mse = mean_squared_error(Y, Ypredict)

    return (Ypredict, r2, mse)


def plot_metrics(X, Y , ylabel, objective):####已知XY绘图找最小/大值
    with plt.style.context('ggplot'):
        plt.plot(X, np.array(Y), '-v', color='blue', mfc='blue')
        if objective=='min':
            idx = np.argmin(Y)
        else:
            idx = np.argmax(Y)
        plt.plot(X[idx], np.array(Y)[idx], 'P', ms=10, mfc='red')

        plt.xlabel('Pixels')
        plt.xticks = X
        plt.ylabel(ylabel)

    plt.show()
    return X[idx]


def find_selected_spec(wl_path,origin_path,percision):#BPNN后已知选择波长，找光谱
    ######输入选择出来的波长（保留小数点后percision位）、随便一个原始谱，输出波长跟光谱
    
    wl,spec = ft.Load_Spectral_Df(origin_path)
    select_wl = np.loadtxt(wl_path)
    select_wl = np.round(np.sort(select_wl),percision)
    wl = np.round(wl,percision)
    select_spec = spec[np.isin(wl,select_wl)]
    select_wl = select_wl[np.isin(select_wl,wl)]

    return np.vstack((select_wl,select_spec)).T

def SD_calculate(spec,n):#算SD
    #导入对应一个波长的所有光谱数据，平均将光谱分为n份的数量
    X_ave = [0 for _ in range(n)]
    every_ave_num = (len(spec)//n)
    sample = np.random.choice(a = every_ave_num*n,size = every_ave_num*n,replace=False,p=None)#无放回抽样
    for i in range(n):
        x_sum = 0
        list_num = i * every_ave_num
        for j in range(list_num,list_num + every_ave_num):
            x_sum += spec[sample[j]]
        X_ave[i] = x_sum/every_ave_num
    SD = np.sqrt(((X_ave - np.mean(X_ave))**2).sum()/(n-1))
    return SD

def RSD_calculate(spec,n):#算RSD
    #导入对应一个波长的所有光谱数据，平均将光谱分为n份的数量
    X_ave = [0 for _ in range(n)]
    every_ave_num = (len(spec)//n)
    sample = np.random.choice(a = every_ave_num*n,size = every_ave_num*n,replace=False,p=None)#无放回抽样
    for i in range(n):
        x_sum = 0
        list_num = i * every_ave_num
        for j in range(list_num,list_num + every_ave_num):
            x_sum += spec[sample[j]]
        X_ave[i] = x_sum/every_ave_num
    RSD = np.sqrt(((X_ave - np.mean(X_ave))**2).sum()/(n-1)) / np.mean(X_ave)
    return RSD

def make_feat(load_path):#制作lable
    lable = pd.read_csv(load_path,sep = ',',header=None).to_numpy()
    lable_iter = 0
    pos = np.where(lable == 'number')[0][0]
    lab_num = lable[pos][1:].astype(int)
    feat = np.zeros((np.sum(lab_num),pos))
    for length,j in zip(lab_num,range(len(lab_num))):
        for i in range(pos):
            feat[lable_iter:(lable_iter + length), i ] = lable[i,j+1]
        lable_iter += length
    return feat


def sgn(num):#滤波
    if(num > 0.0):
        return 1.0
    elif(num == 0.0):
        return 0.0
    else:
        return -1.0

def wavelet_noising(spec):#小波基减背景
    spec = spec.T.tolist()  # 将np.ndarray()转为列表
    w = pywt.Wavelet('sym8')#选择sym8小波基
    [ca5, cd5, cd4, cd3, cd2, cd1] = pywt.wavedec(spec, w, level=5)  # 5层小波分解

    length0 = len(spec)

    Cd1 = np.array(cd1)
    abs_cd1 = np.abs(Cd1)
    median_cd1 = np.median(abs_cd1)

    sigma = (1.0 / 0.6745) * median_cd1
    lamda = sigma * math.sqrt(2.0 * math.log(float(length0 ), math.e))#固定阈值计算
    usecoeffs = []
    usecoeffs.append(ca5)  # 向列表末尾添加对象

    #软硬阈值折中的方法
    a = 0.5
    for cds in [cd1, cd2, cd3, cd4, cd5] :
        for i in range(len(cds)):
            if (abs(cds[i]) >= lamda):
                cds[i] = sgn(cds[i]) * (abs(cds[i]) - a * lamda)
            else:
                cds[i] = 0.0

        usecoeffs.append(cds)
    recoeffs = pywt.waverec(usecoeffs, w)#信号重构
    return recoeffs


def Peak_integrate(wavelength,spec,aim_peak,tol = 1):#已知峰，寻峰面积
    #aim_peak 是一个list或者array
    p = []
    m = []
    if tol == 0:
        return [1 for _ in range(len(aim_peak))],[0 for _ in range(len(aim_peak))],np.where(np.isin(wavelength,aim_peak))[0]
    if isinstance(aim_peak,list):
        aim_peak_arr = np.asarray(aim_peak)
        return Peak_integrate(wavelength,spec,aim_peak_arr,tol)
    elif isinstance(aim_peak,np.ndarray):
        WL_list = np.where(np.isin(wavelength,aim_peak))[0]
        for i in range(len(WL_list)):
        #设定容错为1寻找最小值
            p.append(step('right',spec,tol,WL_list[i]))
            m.append(step('left',spec,tol,WL_list[i]))
       
        return p,m,WL_list
    else:
        print('aim_peak must be a list or array')
        return
    
def step(direct,spec,step,*peak):#寻路
    peak = peak[0]
    if direct == 'right' :
        i = peak
        origin_val = spec[peak]
        min_val = spec[peak+1]
        while  origin_val != min_val:
            if spec[peak] <= spec[peak + 1] :
                peak = peak + 1 
                i = peak
            else:
                i = i + step
                origin_val = min_val
                min_val = np.min(spec[peak+1:i+2])
        return (np.where(np.isin(spec[peak:i+2],min_val))[0][0] + 1)
    elif direct == 'left' :  
        i = peak
        origin_val = spec[peak]
        min_val = spec[peak-1]
        while  origin_val != min_val:
            if spec[peak] <= spec[peak-1]:
                peak = peak - 1
                i = peak
            else:
                i = i - step
                origin_val = min_val
                min_val = np.min(spec[i-1:peak])
                slide = spec[i:peak]
        return (np.where(np.isin(slide[::-1],min_val))[0][0]+1)        
    else:
        print('no direction in Peak_integrate')
        return   


def Anomalous_spectrum_removal(spec):#去异常值
    #输入spec为（波长，谱数）  eg：（40995，92）
    mean_intensity=np.mean(spec,axis = 1)
    max_intensity=np.max(mean_intensity)
    removed = np.all(spec[np.isin(mean_intensity,max_intensity),:] > (max_intensity/2),axis=0)#去除太小的光谱
    return spec[:,removed]


def Normalization(spec,method='total_intensity',peak_para = []):#归一化
    if method=='total_intensity':
        spec=np.around((spec/np.sum(spec)*np.shape(spec)[0]),3)

    elif method=='max_intensity':
        spec=spec/np.max(spec)

    elif method =='internal_standard':#**kw = isin(wavelength,peak_list)
        rel_square = 0

        for i in range(len(peak_para[0])):
            square = np.sum((spec[(peak_para[2][i] - peak_para[1][i]):(peak_para[2][i] + peak_para[0][i])]))
            rel_square += square / ((peak_para[1][i] + peak_para[0][i]))
        spec =  spec / rel_square
        
    return spec

def spec_cut_out(spec,wavelength,start,end,**kw):#直接slice就行，没用的函数。。。
    start_channel=np.argmin(np.abs(wavelength-start))
    end_channel=np.argmin(np.abs(wavelength-end))
    spec=spec[start_channel:end_channel]
    wavelength=wavelength[start_channel:end_channel]
    return wavelength,spec

def spectral_cut_out(spec,wavelength,start,end,**kw):#同上
    start_channel=np.argmin(np.abs(wavelength-start))
    end_channel=np.argmin(np.abs(wavelength-end))
    spec=spec[start_channel:end_channel,:]
    wavelength=wavelength[start_channel:end_channel]
    return wavelength,spec

def select_mean_spectrum(spectral,mean_times,amount):#抽样平均1
    length_of_spec=np.shape(spectral)[0]
    num_of_spec=np.shape(spectral)[1]
    spectral_after_mean=np.zeros((length_of_spec,amount))
    for i in range(amount):
        random_list = random.sample(range(0,num_of_spec),mean_times)
        spectral_after_mean[:,i]=np.sum(spectral[:,random_list],axis=1)/amount  
    return spectral_after_mean

def spec_slice(p,m,aim_peak,wl,spec):#已知峰值的光谱整体切片，用来制作feat
    slice_spec = []
    slice_wl = []
    for single_peak,i in zip(aim_peak,range(len(aim_peak))):#便利每个峰位置的波长切片
        slice_cache = spec[single_peak - m[i]:single_peak + p[i]]
        wl_cache = wl[single_peak - m[i]:single_peak + p[i]]
        #slice_spec.append(slice_cache[0])
        for j in range(len(slice_cache)):
            slice_spec.append(slice_cache[j])
            slice_wl.append(wl_cache[j])
            
    slice_spec = np.array(slice_spec)#convert for saving
    slice_wl = np.array(([slice_wl])).T
    return slice_wl,slice_spec

def spec_slice_feat(p,m,aim_peak,spec):#同上，但是制作同文件中feat

    for single_peak,i in zip(aim_peak,range(len(aim_peak))):#便利每个峰位置的波长切片
        slice_cache = spec[single_peak - m[i]:single_peak + p[i]]
        for j in range(len(slice_cache)):
            slice_spec.append(slice_cache[j])
            
    slice_spec = np.array(slice_spec)#convert for saving

    return slice_spec

def random_choice(start,end,spec_num,mean_num,all_feat):
    random_list = [i for i in range(start,end)]
    for i in range(0,spec_num):    
        random_index = np.random.choice(random_list,mean_num)
        if i == 0:
            mean_feat = np.mean(all_feat[random_index],axis = 0)
        else:
            mean_feat = np.vstack((mean_feat,np.mean(all_feat[random_index],axis = 0)))
    return mean_feat
