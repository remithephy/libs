'''
:@Author: Remi
:@Date: 2023/9/13 08:21:53
:@LastEditors: Remi
:@LastEditTime: 2023/9/13 08:21:53
:Description: 
'''

import numpy as np
import pandas as pd


path = r'D:\20230926\1\100ave.csv'
save_path = r'D:\20230926'
all_lab = pd.read_csv(path, header=None)
all_lab = all_lab.drop(all_lab.index[[1,2,3,4,5]])
all_lab = all_lab.drop(all_lab.index[[0]])

np.savetxt(save_path +'\\' + '1.txt',all_lab,delimiter=';',fmt='%.06f')
#filename = all_lab.iloc[0].tolist()
#all_lab = all_lab.drop(all_lab.index[[0]])
#wavelength = all_lab.iloc[:,0].values
#all_lab.drop(columns=[0],inplace=True)
#for i in range(len(filename)-1):
#    spec = all_lab.iloc[:,i].values
#    np.savetxt(save_path +'\\'+filename[i+1] + '.txt',np.hstack((wavelength,spec)),delimiter=';',fmt='%.06f')