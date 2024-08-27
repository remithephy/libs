'''
:@Author: Remi
:@Date: 2023/8/30 11:02:38
:@LastEditors: Remi
:@LastEditTime: 2023/8/30 11:02:38
:Description: 
'''
import numpy as np
import preprocessing as pp


select_path = r'D:\20230816\数据汇总\新LTB\ML\select_wave_20.csv'
feature_path = r'D:\20230816\数据汇总\新LTB\origin\sliced\2.csv_sliced.csv_ave.csv'
a = pp.find_selected_spec(select_path,feature_path,4)

np.savetxt(r'D:\20230816\数据汇总\新LTB\ML\select_spec_20.csv',a,delimiter=',',fmt='%.06f')