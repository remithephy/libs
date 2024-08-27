'''
:@author: 12184
:@Date: Created on Thu Apr 20 15:32:54 2023
:@LastEditors: Remi
:@LastEditTime: 2023/12/29 15:31:09
:Description: 
'''
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

def find_file(dirpath,postfix):
    
    sample_list = []
    name_list = []
    path_list = []

    path = os.walk(dirpath)
    for dir,fn,ls in path:
        sample_list.append(dir)
        name_list.append(fn)

    sample_list = sample_list[1:]
    name_list = name_list[0]

    for samples in sample_list:
        for files in os.listdir(samples):
            if files.endswith(postfix):
                path_list.append(samples+'\\'+files) 
    return path_list

def read_ava(path):
    all_lab = pd.read_csv(path, header=None ,low_memory=False)
    all_lab = all_lab.drop(all_lab.index[[1,2,3,4,5]])
    all_lab = all_lab.drop(all_lab.index[[0]])
    wavelength = np.array(([all_lab.iloc[:,0].values.astype(float)])).T
    all_lab.drop(columns=[0],inplace=True)
    spec = all_lab.values.astype(float)
    return wavelength,spec

def Load_MultiSpec(spec_filename):
    #导入一行多列的光谱文件
    data = pd.read_csv(spec_filename,header=None,sep = ',')
    wavelength = data.iloc[:,0].values
    data.drop(columns=[0],inplace=True)
    spec = data.values
    return wavelength,spec

def Load_Spectral_Df(spec_filename):
    #用df读
    data = pd.read_csv(spec_filename,header=None,sep = ';')##############################################################
    wavelength=data.iloc[:,0].values
    spec=data.iloc[:,1].values
    return  wavelength,spec

def Load_Spectral(spec_filename):
    # 输入对应的光谱地址,导入光谱文件.
    data=np.loadtxt(spec_filename)
    wavelength=data[:,0]
    spec=data[:,1]
    return wavelength,spec

def Save_Spectral(spec_filename,wavelength,spec):
    # 输入对应的地址与文件名,通道波长,光谱,将光谱保存为asc文件.
    np.savetxt(spec_filename,np.stack((wavelength,spec),axis=1),delimiter=' ',fmt='%.04f')
    return 0

def Load_All_Spectral_in_File(dirpath):
    # 输入对应的文件夹,导入所有光谱文件.
    name_list=[]
    if not dirpath:
        dirpath = os.getcwd()
        #默认情况从当前路径开始
    mylist= os.listdir(dirpath)
        #取出来所有文件和文件夹
    for name in mylist:
        #这里只是一个相对路径
        name_list.append(dirpath+'\\'+name) 
        name = os.path.join(dirpath,name) 
        
    wavelength,spec=Load_Spectral(name_list[0])
    spectral=np.zeros((np.shape(spec)[0],np.shape(name_list)[0]))
    for name,i in zip(name_list,range(np.shape(name_list)[0])):
        wavelength,spectral[:,i]=Load_Spectral(name)      
    return wavelength,spectral,mylist

def Load_All_Spectral_in_File_Df(dirpath):
    # 输入对应的文件夹,导入所有光谱文件.
    name_list=[]
    if not dirpath:
        dirpath = os.getcwd()
        #默认情况从当前路径开始
    mylist= os.listdir(dirpath)
        #取出来所有文件和文件夹
    for name in mylist:
        #这里只是一个相对路径
        if name.endswith(".txt"):#################################################################################这里要改
            name = os.path.join(dirpath,name)
            name_list.append(name)    
    wavelength,spec=Load_Spectral_Df(name_list[0])
    spectral=np.zeros((np.shape(spec)[0],np.shape(name_list)[0]))
    for name,i in zip(name_list,range(np.shape(name_list)[0])):
        wavelength,spectral[:,i]=Load_Spectral_Df(name)      
    return wavelength,spectral,name_list

def Load_Spectral_of_All_Samples(dirpath):
    # 输入对应的目录,导入目录下所有文件夹下所有光谱文件.按照文件夹返回tuple
    name_list=[]
    if not dirpath:
        dirpath = os.getcwd()
        #默认情况从当前路径开始
    mylist= os.listdir(dirpath)
        #取出来所有文件和文件夹
    #os.path.isdir() 判断文件是否是路径
    for name in mylist:
        #这里只是一个相对路径
        name_list.append(dirpath+'\\'+name) 
        name = os.path.join(dirpath,name)    
    samples=[]
    for name,sample_name in zip(name_list,mylist):
        wavelength,spec,spec_list=Load_All_Spectral_in_File(name)
        sample=[sample_name,wavelength,spec,spec_list]
        samples.append(sample)
    return samples    

def Load_Spectral_of_All_Samples_Df(dirpath):
    # 输入对应的目录,导入目录下所有文件夹下所有光谱文件.按照文件夹返回tuple
    name_list=[]
    if not dirpath:
        dirpath = os.getcwd()
        #默认情况从当前路径开始remi/file_tool.py
    mylist= os.listdir(dirpath)
        #取出来所有文件和文件夹
    #os.path.isdir() 判断文件是否是路径
    for name in mylist:
        #这里只是一个相对路径
        name_list.append(dirpath+'\\'+name) 
        name = os.path.join(dirpath,name)    
    samples=[]
    for name,sample_name in zip(name_list,mylist):
        wavelength,spec,spec_list=Load_All_Spectral_in_File_Df(name)
        sample=[sample_name,wavelength,spec,spec_list]
        samples.append(sample)
    return samples    

def Save_All_Spectral_in_File(savepath,wavelength,name_list):
    #将所有的光谱保存到目标文件夹下
    folder=savepath
    if not os.path.isdir(folder):
        os.mkdir(folder)       
    for name,i in zip(name_list,range(np.shape(name_list)[0])):
        Save_Spectral(savepath+'\\'+name,wavelength,spectral[:,i])    
    return 0

def Save_Spectral_of_All_Samples(dirpath,samples):
    #将所有样品的光谱按样品保存到对应的文件夹下
    folder=dirpath
    if not os.path.isdir(folder):
        os.mkdir(folder)       
    for sample in samples:
        savepath,wavelength,spectral,namelist=sample
        Save_All_Spectral_in_File(dirpath+'\\'+savepath,wavelength,spectral,namelist)
    return 0
