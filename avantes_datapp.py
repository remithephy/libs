'''
:@Author: Remi
:@Date: 2023/9/13 08:21:53
:@LastEditors: Remi
:@LastEditTime: 2024/3/17 20:25:23
:Description: 
'''
import os
import numpy as np
import preprocessing as pp
import file_tool as ft

path = r'D:\20240326_fstd'
path_list = ft.find_file(path,'.csv')

filename = [os.path.basename(path_list[i])[:-4] for i in range(len(path_list))]####保留文件名


#################default_iter##############
aim_peak = []
scores = 1
#####ctrl ku  ctrl kc

################################内标用到的峰#######################################
wl,spec = ft.read_ava(path_list[0])
maxspec = np.max(spec,axis = 1)
int_peak = [742.361,744.322,746.852]##氢线656.302      氮线742.361,744.322,746.852     氧777.257
peak_para = pp.Peak_integrate(wl,maxspec,int_peak)



for path_num,paths in enumerate(path_list):
    method = 'total_intensity'#'total_intensity' 'internal_standard'
    print('正在计算：' + paths,'method = ' + method)
    wl,spec = ft.read_ava(paths)
    rsd = []

    channel = 0###########10381最后一个通道，8284倒数第二个spec[8284:,:],进行切片###########

    spec = pp.Anomalous_spectrum_removal(spec[channel:,:])
    wl = wl[channel:]
    spec_num = len(spec[0,:])
    
    rsd.append(pp.RSD_calculate(spec,spec_num))
    #导入对应一个波长的所有光谱数据，平均将光谱分为n份的数量


    ##########weighted method
    ##############genrate slope map#########
    ###################feather devide###########    
    for i in range(spec_num):#calculate slope
        
        spec[:,i] = pp.Normalization(spec[:,i],method,peak_para) 
        
#########################slope 计算部分#########################################  
        single_spec = spec[:,i]
        slope = np.int64((np.append(single_spec,single_spec[-1]) - np.insert(single_spec,0,single_spec[0]))>= 0)#eg  0123 vs 0 -1 1 1 0 ,-1 is the slope between num0&1
        peak = np.insert(slope,0,slope[0]) - np.append(slope,slope[-1])
        if i == 0:
            slope_map = slope
            peak_map = peak
        else:
            slope_map = np.vstack((slope_map,slope)) 
            peak_map = np.vstack((peak_map,peak)) 
            #map 1 = + while 0 = -
    
    slope_map[slope_map == 0] = - 1
    peak_map = peak_map[:,1:-1].T###计算peak形状

    for single_ in slope_map:###计算 arise 和 decay 形状
        for i in range(len(single_)-1):
            if single_[i] != single_[i+1]:
                single_[i] = 0

    scores =  (np.abs(np.sum(peak_map,axis = 1)) +  np.abs(np.sum(slope_map[:,:-1],axis = 0)))/spec_num
###################################################################################

    mean_spec = np.mean(spec,axis = 1)
    wei_spec = (spec.T * scores).T

###########other process method
    '''
    different weighed method
    '''
    #weimean_spec = mean_spec * np.log((scores * (math.e - 1) + 1))
    weimean_spec = mean_spec * scores

    #slice_wl,slice_spec = pp.spec_slice([50,50],[50,50],[11544,11644],wl,spec)
    #slice_wl = slice_wl.flatten()
    weimean_spec = pp.Normalization(weimean_spec,method,peak_para) 
    
    #np.savetxt(path + '//' +filename[path_num]+ 'wei_nm.csv',np.hstack((wl,wei_spec)),delimiter=',',fmt = '%s')
    np.savetxt(path + '//' +filename[path_num]+ 'nm.csv',np.hstack((wl,spec)),delimiter=',',fmt = '%s')

#########for save
    if path_num == 0:
        wl_list = np.hstack((np.array(([['wl']])).T,wl.T))
        mean_spec_list = np.hstack((filename[path_num],mean_spec))
    else:
        mean_spec_list = np.vstack((mean_spec_list,np.hstack((filename[path_num],mean_spec))))
    
    if path_num == 0:
        wl_list = np.hstack((np.array(([['wl']])).T,wl.T))
        wei_spec_list = np.hstack((filename[path_num],weimean_spec))
    else:
        wei_spec_list = np.vstack((wei_spec_list,np.hstack((filename[path_num],weimean_spec))))

#np.savetxt(path + '//' +filename[path_num]+ 'sliced.csv',np.hstack((np.array(([slice_wl])).T,slice_spec)),delimiter=',',fmt='%.04f')
#np.savetxt(path + '//' +filename[path_num]+ 'wei.csv',np.hstack((wl,np.vstack((mean_spec,weimean_spec)).T)),delimiter=',',fmt='%.04f')
#np.savetxt(path + '//' + 'nm.csv',np.vstack((wl_list,mean_spec_list)).T,delimiter=',',fmt = '%s')

#np.savetxt(path + '//' + 'internal_nm.csv',np.vstack((wl_list,mean_spec_list)).T,delimiter=',',fmt = '%s')
#np.savetxt(path + '//' + 'internal_wei.csv',np.vstack((wl_list,wei_spec_list)).T,delimiter=',',fmt = '%s')
