'''
:@Author: Remi
:@Date: 2023/12/17 13:51:30
:@LastEditors: Remi
:@LastEditTime: 2023/12/17 13:51:30
:Description: 
'''
import torch.nn.functional as f
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils import data
import train_fun as tf
import data_prepare as dpp
import time
import preprocessing as pp

T1 = time.time()
##################################数据载入#####################################################
all_lab = pd.read_csv(r'D:\20231017\for_ml\weighted\alllab.csv', header=None).values.astype(np.float32)
all_feat = pd.read_csv(r'D:\20231017\for_ml\weighted\all.csv', header=None).values.astype(np.float32)
all_wave = pd.read_table(r'D:\20231017\preprocessed_for_multi\weighted.csv', header=None, sep=',', usecols=[0]).values.astype(np.float32)
all_wave = all_wave[6:]
perform = []
#cort_index = all_lab[:, 0].argsort()#按顺序排列
#all_lab = all_lab[cort_index, :]
#all_feat = all_feat[cort_index, :]

all_feat_normal = np.zeros(np.shape(all_feat))
T2 = (time.time()-T1)
print('数据读取完成' + 'time = ' , T2)
####################################数据归一化##############################################
for i in range(len(all_feat[:,0])):
    all_feat_normal[i] = pp.Normalization(all_feat[i],method='total_intensity')

T3 = (time.time()-T1)
print('归一化完成' , T3)
#with plt.style.context('ggplot'):
#    for i in range(0,6):
#        plt.plot(all_wave[11500:12200], all_feat[i,11500:12200])#2960:2980
#    plt.xlabel("Pixels")
#    plt.ylabel("intensity")
#    plt.show()
###################################进行随机采样#################################################

label_num = 6 #6个label
spec_num = 60#对每个label出20个谱
mean_num = 40 #10次一平均
label_list = np.zeros(label_num)
avg_lab = np.zeros(label_num * spec_num)
########################make new feat############################
for i in range(label_num):#find label
    label_list[i] = all_lab[int((2*i + 0.5)*len(all_lab)/(2 * label_num))]#int((2*i + 0.9) 在这里添加
    start = np.where(all_lab == label_list[i])[0][0]
    end = np.where(all_lab == label_list[i])[0][-1]

    print(start,end)

    if i == 0:
        avg_feat = pp.random_choice(start,end,spec_num,mean_num,all_feat)#np.mean(all_feat[start:end,:],axis = 0)#
    else:
        avg_feat = np.vstack((avg_feat,pp.random_choice(start,end,spec_num,mean_num,all_feat)))#np.vstack((avg_feat,np.mean(all_feat[start:end,:],axis = 0)))# 
print('label_list = ',label_list)
########################make new label########################
avg_lab = np.array([label_list[i]  for i in range(label_num) for num in range(spec_num)])
###############################光谱平均处理及数据集划分#####################################################
T4 = (time.time()-T1)
print('随机抽样耗时' , T4)

'''
with plt.style.context('ggplot'):
    for i in range(0,100):
        plt.plot(all_wave[11530:11550], avg_feat[i,11530:11550])#2960:2980
    plt.xlabel("Pixels")
    plt.ylabel("intensity")
    plt.show()
    '''


batch_size = label_num*10
avg_feat = avg_feat[:,11500:12200]



for i in range(label_num):
    index_test = [i]
    train_feat, train_lab, test_feat, test_lab = tf.stand_divid(avg_feat, avg_lab, label_num, index_test)
    scores = dpp.score(train_feat, train_lab)
    num_feat = 10
    index_feat = dpp.select_best(scores, num_feat)
    train_feat = train_feat [:, index_feat]
    test_feat = test_feat [:, index_feat]
    T5 = (time.time()-T1)
    print('划分选择耗时' , T5)
    train_feat = tf.a2t(train_feat)
    train_lab_tensor = tf.a2t(train_lab)
    test_feat = tf.a2t(test_feat)
    test_lab_tensor = tf.a2t(test_lab)
    train_data = tf.load_array((train_feat, train_lab_tensor), batch_size)
    ##############################训练##########################################################

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    index_test = [i]
    print('模型训练开始')
    model = tf.BPNN(n_feature=num_feat, n_hidden=5, n_output=1)
    model,rtsta = tf.small_model_train(model, train_data, train_feat, train_lab_tensor, test_feat, test_lab_tensor, device, logdir)
    perform.append(rtsta)
    print('模型训练结束')
print(perform)


with plt.style.context('ggplot'):
    plt.plot(all_wave[11500:12200], scores.T, color='b')
    plt.scatter(all_wave[11500:12200][index_feat], scores[index_feat], color='red')
    plt.xlabel("Pixels")
    plt.ylabel("intensity")
    
    plt.show()