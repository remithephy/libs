
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

max_path = r'D:\20230911\maxint.xlsx'
ave_path = r'D:\20230911\integrate.xlsx'
noise_path = r'D:\20230911\noise.xlsx'

max_data = pd.read_excel(max_path, header=0,index_col= 0 )
ave_data = pd.read_excel(ave_path, header=0,index_col= 0 )
noise_data = pd.read_excel(noise_path, header=0,index_col= 0 )
SNR = 20 * np.log10(ave_data/noise_data)
weighted_SNR = SNR * max_data

cmap=mpl.cm.RdYlBu_r#获取色条
newcolors=cmap(np.linspace(0,1,256))#分片操作
newcmap=ListedColormap(newcolors[100:])#只取100之后的颜色列表，前面的舍去

sns.set_theme(font_scale=1.5)
sns.set_context({"figure.figsize":(8,8)})
sns.heatmap(data=max_data,square=True, cmap=newcmap,vmin=800, vmax=2500).set_title('max')
#sns.heatmap(data=noise_data,square=True, cmap=newcmap).set_title('noise')
#sns.heatmap(data=SNR,square=True, cmap=newcmap).set_title('SNR')

plt.show()