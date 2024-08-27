# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 14:27:46 2021

@author: 张朝
"""

import pandas as pd
import numpy as np
import re
import os
import math
import random
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#################计算RMSE#################################
def compute_rmse(y,y_hat):
    return math.sqrt(((y_hat-y)**2).mean())
#################cca特征选择#################################
def select_feat_cca(feat, lab, num_feat):
    from sklearn.cross_decomposition import CCA
    cca = CCA(n_components=1)
    cca = cca.fit(feat, lab)
    coff = cca.coff_
    coff = abs(coff)
    best_index = np.argsort(coff)[-num_feat:, :]
    best_index = best_index.ravel()
    select_feat = feat[:, best_index]
    return select_feat


#################pls特征选择#########################################
def select_feat_pls(feat, lab, num_feat):
    from sklearn.cross_decomposition import PLSRegression
    pls = PLSRegression(n_components=num_feat)
    pls = pls.fit(feat, lab)
    select_feat = pls.predict(feat)
    return select_feat


##################corr特征选择#############################################
def select_feat_corr(feat, lab, num_feat):
    from sklearn.feature_selection import SelectKBest, f_regression
    skb = SelectKBest(score_func=f_regression, k=num_feat)
    select_feat = skb.fit_transform(feat, lab)
    return select_feat


##################chi2特征选择###################################################
def select_feat_chi2(feat, lab, num_feat):
    from sklearn.feature_selection import SelectKBest, chi2
    skb = SelectKBest(score_func=chi2, k=num_feat)
    select_feat = skb.fit_transform(feat, lab)
    return select_feat


###########线性拟合函数##########################################################
def func_linear(x, k, b):
    return k * x + b


###########单变量模型##########################################################
def univariate_model(feat, lab, fit_func):
    p, c = curve_fit(fit_func, feat, lab)
    return p


###########窗口光谱平均##########################################################
def spec_avg_windows(feat, length_wind, step_avg, num_sample):
    num_spec = len(feat) // num_sample
    num_avg = (num_spec - length_wind) // step_avg + 1
    avg_feat = []
    for i in range(num_sample):
        for j in range(num_avg):
            avg_feat.append(feat[i * num_spec + j * step_avg:i *
                                                             num_spec + j * step_avg + length_wind, :].mean(0))
    avg_feat = np.array(avg_feat)
    avg_feat = avg_feat.reshape(int(num_avg * num_sample), -1)
    return avg_feat


###########普通光谱平均##########################################################
def spec_avg_normal(feat, step_avg):
    num_spec = len(feat) // step_avg
    avg_feat = []
    for i in range(num_spec):
        avg_feat.append(feat[i * step_avg:(i + 1) * step_avg, :].mean(0))
    avg_feat = np.array(avg_feat)
    avg_feat = avg_feat.reshape(num_spec, -1)
    return avg_feat


###########读取大型csv文件##########################################################
def load_big_csv(file_path, chunk_size):
    df_chunk = pd.read_csv(file_path, header=None, chunksize=chunk_size)
    list_chunk = []
    for chunk in df_chunk:
        list_chunk.append(chunk)
    df_concat = pd.concat(list_chunk)
    df_concat = np.array(df_concat).astype(np.float32)
    return df_concat


###########获取谱线的中心强度##########################################################
def get_line_inty(all_feat, waves, wave_line, step_wave=0.02):
    waves = waves.ravel()
    all_line_inty = []
    for i in range(len(wave_line)):
        spec_index = np.argwhere(
            (waves > wave_line[i] - step_wave) & (waves < wave_line[i] + step_wave))
        spec_index = spec_index.ravel()
        select_spec = all_feat[:, spec_index]
        inty = np.max(select_spec, axis=1)
        all_line_inty.append(inty)
    all_line_inty = np.array(all_line_inty)
    all_line_inty = all_line_inty.T
    return all_line_inty


###########洛伦兹函数定义##########################################################
def lorz_func(x, y0, A, xc, w):
    y = y0 + (2 * A / np.pi) * (w / (4 * (x - xc) ** 2 + w ** 2))
    return y


###########对谱线进行洛伦兹拟合######################################################
def lorz_fit(x, y, func, p0):
    p, c = curve_fit(func, x, y, p0)
    y0, A, xc, w = p
    yc = y0 + (2 * A) / (w * np.pi)

    return np.array([xc, yc, w])


###########获取均值和标准差######################################################
def get_sd_mean(train_lab, train_pred_lab, test_lab, test_pred_lab, num_sample):
    # 将训练集和测试集放在一起了
    num_spec = (len(train_lab) + len(test_lab)) // num_sample
    train_lab = train_lab.reshape(-1, num_spec)
    train_pred_lab = train_pred_lab.reshape(-1, num_spec)
    test_lab = test_lab.reshape(-1, num_spec)
    test_pred_lab = test_pred_lab.reshape(-1, num_spec)
    train_pred_sd = np.std(train_pred_lab, axis=1, ddof=1)
    test_pred_sd = np.std(test_pred_lab, axis=1, ddof=1)
    train_pred_mean = np.mean(train_pred_lab, axis=1)
    test_pred_mean = np.mean(test_pred_lab, axis=1)
    train_mean = np.mean(train_lab, axis=1)
    test_mean = np.mean(test_lab, axis=1)
    # 将训练集和测试集的标签，预测，标准差拼接在一起了，形成列数组
    mean = np.concatenate((train_mean, test_mean))
    pred_mean = np.concatenate((train_pred_mean, test_pred_mean))
    pred_sd = np.concatenate((train_pred_sd, test_pred_sd))
    mean = mean.reshape(-1, 1)
    pred_mean = pred_mean.reshape(-1, 1)
    pred_sd = pred_sd.reshape(-1, 1)
    all_sd_mean = np.concatenate((mean, pred_mean, pred_sd), axis=1)
    return all_sd_mean


############绘制校准曲线#####################################################
def plot_errbar_curve(train_lab, train_pred_lab, test_lab, test_pred_lab, num_sample):
    num_spec = (len(train_lab) + len(test_lab)) // num_sample

    train_lab = train_lab.reshape(-1, num_spec)
    train_pred_lab = train_pred_lab.reshape(-1, num_spec)
    test_lab = test_lab.reshape(-1, num_spec)
    test_pred_lab = test_pred_lab.reshape(-1, num_spec)
    train_pred_sd = np.std(train_pred_lab, axis=1, ddof=1)
    test_pred_sd = np.std(test_pred_lab, axis=1, ddof=1)
    train_pred_mean = np.mean(train_pred_lab, axis=1)
    test_pred_mean = np.mean(test_pred_lab, axis=1)
    train_mean = np.mean(train_lab, axis=1)
    test_mean = np.mean(test_lab, axis=1)
    #################绘制误差棒图像########################
    fig = plt.figure()
    plt.errorbar(train_mean, train_pred_mean, yerr=train_pred_sd, label='train data', capsize=5, linestyle='none',
                 marker='.', ecolor='red', ms=10)
    plt.errorbar(test_mean, test_pred_mean, yerr=test_pred_sd, label='test data', capsize=5, linestyle='none',
                 marker='.', ecolor='red', ms=10)

    x = np.linspace(min(train_mean), max(train_mean), 2)
    y = x
    plt.plot(x, y)
    plt.grid()
    plt.xlabel("Reference Co.(um)", fontsize=16)
    plt.ylabel("Predict Co.(um)", fontsize=16)
    plt.legend(fontsize=16)
    return fig


############计算相对误差#####################################################
def compute_MRE(y, y_hat):
    rme = (np.abs((y_hat - y) / y)).mean()
    return rme


############确定最佳的epoch#####################################################
def optim_epoch(test_loss, num_epochs, step_epoch):
    test_loss_mean = []
    num = num_epochs // step_epoch
    for i in range(num - 3):
        test_loss_mean.append((sum(test_loss[i:i + 3])) / 3.0)
    best_epoch = test_loss_mean.index(min(test_loss_mean[60:])) + 1
    print(test_loss_mean)
    return best_epoch


############计算rsd#####################################################
def compute_rsd(y_hat, test_num):
    y_hat = y_hat.reshape(test_num, -1)
    rsd = (y_hat.std(axis=1, ddof=1) / y_hat.mean(1)).mean()

    return abs(rsd)


############计算r平方#####################################################
def compute_r2(y, y_hat):
    y = y.flatten()
    y_hat = y_hat.flatten()
    a1 = ((y_hat - y) ** 2).sum()
    a2 = ((y - y.mean()) ** 2).sum()
    r_2 = 1 - a1 / a2
    return r_2


############计算斜率#####################################################
def compute_slope(y, y_hat):
    y = y.flatten()
    y_hat = y_hat.flatten()
    coff = np.polyfit(y, y_hat, 1)
    return coff[0]


############加权的相关性挑选特征#####################################################
def score_weighted(x, y):
    scores = score(x, y)
    # scores=scores*scores
    ints_avg = x.mean(0) * 12
    scores = scores + ints_avg
    return scores


############产生标签###########################################################
def get_lab(lab_sample, num_spec, num_sample):
    n = int(num_spec)
    all_lab = np.ones([num_spec * num_sample, 1])
    for i in range(num_sample):
        all_lab[i * n:(i + 1) * n, :] = lab_sample[i]
    return all_lab


############读取所有txt数据并保存为CSV文件###########################################################
def load_txt(txt_path, csv_path):
    all_data = []
    all_lab = []
    for root, dirs, files in os.walk(txt_path):
        for file in files:
            if '.txt' in os.path.join(root, file):
                data = pd.read_table(os.path.join(
                    root, file), sep=';', header=None, usecols=[1]).values
                data_T = data.T
                data_T = data_T.ravel()
                all_data.append(data_T)
                all_lab.append(os.path.join(root, file))
    all_data = np.array(all_data)
    pd.DataFrame(all_data).to_csv(
        csv_path + '/all_feat.csv', header=None, index=False)
    return all_data, all_lab


############光谱数据平均###########################################################
def avg_data(spec_data, n, is_overlop=0, m=0, n_sample=0):
    # n:平均窗口长度
    # is_overlop:是否需要重叠
    # m:窗口移动的步长
    # n_sample:样品的个数
    avg_specs = []
    n_spec = len(spec_data)

    if is_overlop == 0:
        k = n_spec // n
        for i in range(k):
            avg_spec = (spec_data[i * n:(i + 1) * n, :]).mean(0)
            avg_specs.append(avg_spec)

    else:
        sample_spec = n_spec // n_sample
        k = ((sample_spec - n) // m) + 1
        for i in range(int(n_sample)):
            for j in range(k):
                avg_spec = (
                    spec_data[i * sample_spec + j * k:i * sample_spec + j * k + n, :]).mean(0)
                avg_specs.append(avg_spec)

    avg_specs = np.array(avg_specs)
    return avg_specs


############多项式基线扣除#####################################################
def poly_backcorr(y, degree, m, thres):
    # y:raw data
    # m:分的段数
    # n:多项式阶数
    # thres:偏差阈值
    length = len(y) // m
    y_sub = []
    for i in range(m):
        y_sub.append(y[i * length:(i + 1) * length])
    y_sub = np.array(y_sub)

    x = range(length)
    x = np.array(x) + 1
    num = 10
    baseline_y = []
    sum_loss = 0
    ret = 0
    for i in range(len(y_sub)):
        for j in range(num):
            fit_f = np.polyfit(x, y_sub[i, :], degree)
            p1 = np.poly1d(fit_f)
            y_vals = p1(x)
            loss = (np.abs(y_sub[i, :] - y_vals)).sum()
            if sum_loss < thres:
                ret = 1

            if i == num - 1 or ret == 1:
                baseline_y.append(y_vals)
                break
            for k in range(len(x)):
                if y_vals[k] <= y_sub[i, k]:
                    y_sub[j, k] = y_vals[k]

            # plt.plot(x,y)
    baseline_y = np.array(baseline_y)
    baseline_y = baseline_y.ravel()
    corct_y = y - baseline_y
    #for i in range(len(corct_y)):
    #    if corct_y[i] < 0:
    #        corct_y[i] = 0
    return corct_y


############模型评估###########################################################
def model_evlate(y, y_hat):
    a1 = ((y_hat - y) ** 2).sum()
    a2 = ((y - y.mean()) ** 2).sum()
    r_squre = 1 - a1 / a2
    a3 = np.sqrt(a2)
    a4 = np.sqrt(((y_hat - y_hat.mean()) ** 2).sum())
    a5 = ((y - y.mean()) * (y_hat - y_hat.mean())).sum()
    r_p = a5 / (a3 * a4)
    rme = (np.abs((y_hat - y) / y)).mean()
    rmse = (np.sqrt((y_hat - y) ** 2)).mean()
    rsd = np.sqrt(y_hat.var(1)) / y_hat.mean(1)
    coff = np.polyfit(y, y_hat, 1)
    slope = coff[0]
    return r_squre, slope, rme, rsd


############交叉验证###########################################################
def cross_vali(feat, lab, n):
    train_lab = []
    vali_lab = []
    train_feat = []
    vali_feat = []
    num = len(lab) / n
    num = int(num)
    for i in range(n):
        vali_lab.append(lab[i * num:num * (i + 1), :].reshape(-1, 1))
        vali_feat.append(feat[i * num:num * (i + 1), :])

        train_lab.append(np.delete(lab, range(i * num, num * (i + 1)), axis=0))
        train_feat.append(
            np.delete(feat, range(i * num, num * (i + 1)), axis=0))
    return train_feat, train_lab, vali_feat, vali_lab


############计算每个像素的rsd#####################################################
def rsd_pixel(feat, sample_num):
    spec_num = len(feat) // sample_num
    rsd_pixel = []
    for i in range(sample_num):
        sample_std = np.std(
            feat[i * spec_num:(i + 1) * spec_num, :], axis=0, ddof=1)
        sample_mean = np.mean(feat[i * spec_num:(i + 1) * spec_num, :], axis=0)
        rsd_pixel.append(sample_std / sample_mean)
    return rsd_pixel


############直接读取数据#####################################################
def read_data():
    file_addres = r"C:\Users\张朝\Desktop\工作\数据\20210714原始光谱数据"
    a = "\\"
    all_data = pd.DataFrame()
    num_data = 62
    all_labels = []

    for file_name in os.listdir(file_addres):
        file_name = re.match(".{2,3}um", file_name)
        if file_name:
            for i in range(num_data):
                pattern = re.compile("(.+)um")
                result = pattern.findall(file_name.string)
                all_labels.append(float(result[0]))
            for data_name in os.listdir(file_addres + a + file_name.string):
                data_name = re.match(".{1,3}txt", data_name)
                if data_name:
                    with open(file_addres + a + file_name.string + a + data_name.string) as in_files:
                        data = pd.read_table(
                            in_files, sep=";", header=None, usecols=[1])
                        data = data.T
                        all_data = all_data.append(data)

    all_data = all_data.values
    all_labels = np.array(all_labels)
    return all_data, all_labels


############光谱平均处理方法1#################################################
def data_mean(k_mean):
    file_addres = os.getcwd()
    a = "\\"
    all_data = pd.DataFrame()
    deal_data = pd.DataFrame()
    num_data = 62
    all_labels = []

    for file_name in os.listdir(file_addres):
        file_name = re.match(".{2,3}um", file_name)
        if file_name:
            for i in range(num_data // 10):
                pattern = re.compile("(.+)um")
                result = pattern.findall(file_name.string)
                all_labels.append(float(result[0]))
            for data_name in os.listdir(file_addres + a + file_name.string):
                data_name = re.match(".{1,3}txt", data_name)
                if data_name:
                    with open(file_addres + a + file_name.string + a + data_name.string) as in_files:
                        data = pd.read_table(
                            in_files, sep=";", header=None, usecols=[1])
                        data = data.T
                        all_data = all_data.append(data)

    all_data = all_data.values
    n = 10
    data_len = all_data.shape[0] // (k_mean * n)

    for i in range(k_mean * n):
        deal_data = deal_data.append(pd.DataFrame(
            all_data[i * data_len:(i + 1) * data_len, :].mean(0).reshape(1, -1)))

    deal_data.to_csv("mean_features" + str(k_mean) +
                     ".csv", index=False, header=False)

    all_labels = np.array(all_labels)
    all_labels = pd.DataFrame(all_labels)
    all_labels.to_csv("deal_labels" + str(k_mean) +
                      ".csv", index=False, header=False)

    return deal_data.values, all_labels


############光谱平均处理方法2################################################
def smooth_data(avg_num, step_leng):
    file_addres = os.getcwd()
    a = "\\"
    all_data = pd.DataFrame()
    deal_data = pd.DataFrame()

    all_labels = []

    for file_name in os.listdir(file_addres):
        file_name = re.match(".{2,3}um", file_name)
        if file_name:
            for data_name in os.listdir(file_addres + a + file_name.string):
                data_name = re.match(".{1,3}txt", data_name)
                if data_name:
                    with open(file_addres + a + file_name.string + a + data_name.string) as in_files:
                        data = pd.read_table(
                            in_files, sep=";", header=None, usecols=[1])
                        data = data.T
                        all_data = all_data.append(data)

    all_data = all_data.values
    sample_num = 10
    spec_num = all_data.shape[0] // sample_num
    n = (spec_num - avg_num) // step_leng + 1

    for i in range(sample_num):
        sample_data = all_data[i * spec_num:(i + 1) * spec_num]
        for j in range(n):
            deal_data = deal_data.append(
                pd.DataFrame(sample_data[j * step_leng:j * step_leng + avg_num, :].mean(0).reshape(1, -1)))
    for file_name in os.listdir(file_addres):
        file_name = re.match(".{2,3}um", file_name)
        if file_name:
            for i in range(n):
                pattern = re.compile("(.+)um")
                result = pattern.findall(file_name.string)
                all_labels.append(float(result[0]))

    deal_data.to_csv("smooth_features" + str(avg_num) +
                     str(step_leng) + ".csv", index=False, header=None)

    all_labels = np.array(all_labels)
    all_labels = pd.DataFrame(all_labels)
    all_labels.to_csv("deal_labels" + str(avg_num) +
                      str(step_leng) + ".csv", index=False, header=None)

    return deal_data.values, all_labels, all_data


###########################根据像素的方差得分#######################################
def pixel_var(x):
    var_x = np.var(x, axis=0)

    return var_x


###########################相关性得分#######################################
def score(x, y):#皮尔逊系数
    var_y = np.var(y)
    mean_y = np.mean(y)
    mean_x = np.mean(x, 0)
    var_x = np.var(x, 0)
    num_pixel = x.shape[1]
    cov = np.zeros(num_pixel)
    corr = np.zeros(num_pixel)
    scores = np.zeros(num_pixel)
    for i in range(x.shape[1]):
        for j in range(x.shape[0]):
            cov[i] = cov[i] + (y[j] - mean_y) * \
                     (x[j, i] - mean_x[i]) / x.shape[0]
    for i in range(x.shape[1]):
        corr[i] = cov[i] / math.sqrt(var_x[i] * var_y)
        scores[i] = (corr[i] ** 2) / (1 - corr[i] ** 2)
    return scores


###########################按照光谱归一化处理#######################################
def normal_spec(x):
    x_min = x.min(1)
    x_max = x.max(1)
    y = np.ones_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            y[i, j] = (x[i, j] - x_min[i]) / (x_max[i] - x_min[i])
    return y


###########################按照像素归一化处理#################################
def normal_pixel(x):
    x_min = x.min(0)
    x_max = x.max(0)
    new_feat=np.ones_like(x)
    for i in range(x.shape[1]):
        for j in range(x.shape[0]):
            new_feat[j, i] = (x[j, i] - x_min[i]) / (x_max[i] - x_min[i])

    return new_feat


#############################删除指定的样品###################################
def delet_sample(x, y, sample_delet, num_sample):
    index_delet = []
    num_spec = len(x) // num_sample
    for i in sample_delet:
        index_delet.append(np.arange(i * num_spec, (i + 1) * num_spec))
    index_delet = np.array(index_delet)
    index_delet = index_delet.ravel()
    x = np.delete(x, index_delet, axis=0)
    y = np.delete(y, index_delet, axis=0)
    return x, y


############################对特征反tan处理################################################
def feat_fantan(x):
    return x


#############################对颗粒尺寸归一化处理##############################
def lab_normal(y):
    y_min = y.min()
    y_max = y.max()
    for i in range(y.shape[0]):
        y[i, 0] = (y[i, 0] - y_min) / (y_max - y_min)
    return y


#############################按照光谱标准化处理########################################
def stand_spec(x):
    mean_x_spec = x.mean(1)
    std_x_spec = np.std(x, axis=1, ddof=1)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = (x[i, j] - mean_x_spec[i]) / std_x_spec[i]
    return x


#############################按照像素标准化处理########################################
def stand_pixel(x):
    mean_x_pixel = x.mean(0)
    std_x_pixel = np.std(x, axis=0, ddof=1)
    for j in range(x.shape[1]):
        for i in range(x.shape[0]):
            x[i, j] = (x[i, j] - mean_x_pixel[j]) / std_x_pixel[j]
    return x


#############################根据相关性挑选特征##########################################
def select_best(scores, n):
    scores_index = scores.argsort(0)
    best_index = scores_index[-n:]
    return best_index.flatten()


#############################标签转换########################################
def lab_trans(lab):
    for i in range(lab.shape[0]):
        if lab[i, 0] >= 109 and lab[i, 0] < 390:
            lab[i, 0] = lab[i, 0] + 170
    return lab
############################################################################
# cd,dc=data_mean(8)
# cd,dc=smooth_data(10,6)
# wave=pd.read_csv(r"C:\Users\张朝\Desktop\工作\数据\20210714原始光谱数据\77um\1.txt",sep=";",header=None,usecols=[0])
# wave.to_csv("wave.csv",header=None,index=False)
