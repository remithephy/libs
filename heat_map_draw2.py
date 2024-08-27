
'''
:@Author: Remi
:@Date: 2023/9/13 08:21:53
:@LastEditors: Remi
:@LastEditTime: 2023/9/13 08:21:53
:Description: 
'''

import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import matplotlib.pyplot as plt

signal_path = r'D:\20240411\mean_spec\signal.csv'
snr_path = r'D:\20240411\mean_spec\SNR.csv'

signal = pd.read_csv(signal_path, header=0,index_col= 0)
snr = pd.read_csv(snr_path, header=0,index_col= 0)


cmap=mpl.cm.RdYlBu_r#获取色条
newcolors=cmap(np.linspace(0,1,256))#分片操作
newcmap=ListedColormap(newcolors[100:])#只取100之后的颜色列表，前面的舍去

plt.figure(1)
sns.set_theme(font_scale=1.5)
sns.set_context({"figure.figsize":(8,8)})
sns.heatmap(data=signal,square=True, cmap=newcmap,vmin=10, vmax=400).set_title('signal')
plt.show()

plt.figure(2)
sns.set_theme(font_scale=1.5)
sns.set_context({"figure.figsize":(8,8)})
sns.heatmap(data=snr,square=True, cmap=newcmap,vmin=0, vmax=25).set_title('snr')
plt.show()