'''
:@Author: Remi
:@Date: 2023/8/7 16:26:18
:@LastEditors: Remi
:@LastEditTime: 2023/8/7 16:26:18
:Description: 
'''
import os
import pandas as pd
import numpy as np

name_list=[]
dirpath = r'D:\20230816\数据汇总\新LTB\normalized'#数据文件夹

mylist= os.listdir(dirpath)

for name in mylist:
    if name.endswith(".txt"):
            name_list.append(dirpath+'\\'+name) 
            name = os.path.join(dirpath,name)
           
for spec_filename in name_list:           
    data = pd.read_csv(spec_filename,sep ='\t')#sep 读入数据的间隔符
    wavelength=data.iloc[:,0].values
    spec=data.iloc[:,1].values
    np.savetxt(spec_filename,np.stack((wavelength,spec),axis=1),delimiter='\t',fmt='%.04f')#sep 存储数据的间隔符
