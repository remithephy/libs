import random
import numpy as np
import pandas as pd

def select_mean_spectrum(spectral,mean_times,amount):#抽样平均1
    length_of_spec=np.shape(spectral)[0]
    num_of_spec=np.shape(spectral)[1]
    spectral_after_mean=np.zeros((length_of_spec,amount))
    for i in range(amount):
        random_list = random.sample(range(0,num_of_spec),mean_times)
        spectral_after_mean[:,i]=np.sum(spectral[:,random_list],axis=1)/amount  
    return spectral_after_mean

def read_ava(path):
    all_lab = pd.read_excel(path, header=None )
    all_lab = all_lab.drop(all_lab.index[[1,2,3,4,5]])
    all_lab = all_lab.drop(all_lab.index[[0]])
    wavelength = np.array(([all_lab.iloc[:,0].values.astype(float)])).T
    all_lab.drop(columns=[0],inplace=True)
    spec = all_lab.values.astype(float)
    return wavelength,spec

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


file_path = r'C:\Users\0\Documents\WeChat Files\wxid_lo02hcn48pud22\FileStorage\File\2024-07'
file_name = 'U44.xlsx'
wl,spec = read_ava(file_path + '\\' + file_name)


random_selected_spec = select_mean_spectrum(spec,10,100)#10个一平均出100个平均后的谱
rsd = [RSD_calculate(random_selected_spec[i,:],len(random_selected_spec[:,0])) for i in range(len(random_selected_spec[0,:]))]

np.savetxt(file_path + '\\' + 'meand' + file_name[:-5] + '.csv',random_selected_spec,delimiter=',')
np.savetxt(file_path + '\\' + 'rsd' + file_name[:-5] + '.csv',rsd,delimiter=',')