import preprocessing as pp
import numpy as np

path = r'D:\20240412\Rb_analysis'


'''
path 文件格式：
la 1 2 3 6 7 9#浓度
Sm 2 4 5 6 7 8
number 192 135 162 165 147 185#谱数量

'''

feat = pp.make_feat(path + '//' + 'makelab.csv')

np.savetxt(path + '//' + 'alllab.csv',feat,delimiter=',',fmt='%.06f')