import numpy as np
import file_tool as ft

import os

namelist = []
spec_num = []
path = r'D:\20231017(仿真卤水)\weighted'#选择原始文件夹

for name in os.listdir(path):#读.csv的文件
    if name.endswith(".csv"):
        namelist.append(path + '\\' + name)
        
print(namelist)
for names in namelist:
#    wl,spec = ft.Load_MultiSpec(names)#LTB
    wl,spec = ft.read_ava(names)
    spec_num.append(len(spec[0]))
    if names == namelist[0]:
        spec1 = spec.T
    else:
        spec1 = np.vstack((spec1,spec.T))


print(namelist)
print(spec_num)

np.savetxt(path + '\\' + 'all.csv',spec1,delimiter=',',fmt='%.06f')