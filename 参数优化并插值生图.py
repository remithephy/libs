'''
:@Author: Remi
:@Date: 2024/4/11 21:18:54
:@LastEditors: Remi
:@LastEditTime: 2024/4/11 21:18:54
:Description: 
'''
import cmaps
import os
import file_tool as ft
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
path = r'D:\20240411\mean_spec'
path_list = ft.find_file(path,'.csv')

filename = [os.path.basename(path_list[i])[:-4] for i in range(len(path_list))]####保留文件名

noise_start = '849.529'
noise_end = '851.228'
aim = '852.075'
wl,spec = ft.Load_MultiSpec(path_list[0])
head_sig = np.hstack(('signal',spec[0,:]))
head_SNR = np.hstack(('SNR',spec[0,:]))
#################default_iter##############
aim_peak = []
scores = 1
#####ctrl ku  ctrl kc
for path_num,paths in enumerate(path_list):
    print(path_num,paths[-10:-8])
    wl,spec = ft.Load_MultiSpec(paths)
    idx_noise = [np.where(np.isin(wl,noise_start))[0][0],np.where(np.isin(wl,noise_end))[0][0]]

    noise = np.mean(spec[idx_noise[0]:idx_noise[1],:],axis = 0)
    signal = spec[np.where(np.isin(wl,aim))[0][0],:]

    ref_signal = signal - noise
    SNR = ref_signal/np.sqrt(noise)

    ref_signal = np.hstack((paths[-10:-8],ref_signal))
    SNR = np.hstack((paths[-10:-8],SNR))


    head_sig = np.vstack((head_sig,ref_signal))
    head_SNR = np.vstack((head_SNR,SNR))
np.savetxt(path + '//' + 'signal.csv',head_sig,delimiter=',',fmt = '%s')
np.savetxt(path + '//' + 'SNR.csv',head_SNR,delimiter=',',fmt = '%s')



#生成插值机
#对SIGNAL
int_num = 21

TD = head_sig[0,1:].astype(float)
Energy = head_sig[1:,0].astype(float)
X,Y = np.meshgrid(TD,Energy)
X = X.reshape(-1)
Y = Y.reshape(-1)
Z = head_sig[1:,1:].astype(float).reshape(-1)

linear_sig = LinearNDInterpolator(list(zip(X,Y)), Z)

int_x = np.linspace(min(TD), max(TD),num = int_num)
int_y = np.linspace(min(Energy), max(Energy),num = int_num)
int_x,int_y = np.meshgrid(int_x,int_y)
int_z = linear_sig(int_x,int_y)

#对SNR
TD2 = head_SNR[0,1:].astype(float)
Energy2 = head_SNR[1:,0].astype(float)
X2,Y2 = np.meshgrid(TD2,Energy2)
X2 = X2.reshape(-1)
Y2 = Y2.reshape(-1)
Z2 = head_SNR[1:,1:].astype(float).reshape(-1)

linear_SNR = LinearNDInterpolator(list(zip(X2,Y2)), Z2)

int_x2 = np.linspace(min(TD2), max(TD2),num = int_num)
int_y2 = np.linspace(min(Energy2), max(Energy2),num = int_num)
int_x2,int_y2 = np.meshgrid(int_x2,int_y2)
int_z2 = linear_SNR(int_x2,int_y2)

from matplotlib.colors import ListedColormap
cmap = cmaps.cmp_b2r
newcolors=cmap(np.linspace(0,1,256))
newcmap=ListedColormap(newcolors[::1])

#draw
plt.figure(1)
plt.title('Signal')
plt.pcolormesh(int_x, int_y, int_z, shading='auto',cmap = newcmap)
plt.plot(X, Y, "ok", label="measured")
plt.xlabel('TD(us)')
plt.ylabel('Energy(mJ)')
plt.legend(loc='upper left')
plt.colorbar()
plt.show()

plt.figure(2)
plt.title('SNR')
plt.pcolormesh(int_x2, int_y2, int_z2, shading='auto')
plt.plot(X2, Y2, "ok", label="measured")
plt.xlabel('TD(us)')
plt.ylabel('Energy(mJ)')
plt.legend(loc='upper left')
plt.colorbar()
#plt.show()

np.savetxt(path + '//' + 'intSNR.csv',np.vstack((int_x.reshape(-1),int_y.reshape(-1),int_z.reshape(-1))).T,delimiter=',')
np.savetxt(path + '//' + 'intsignal.csv',np.vstack((int_x2.reshape(-1),int_y2.reshape(-1),int_z2.reshape(-1))).T,delimiter=',')