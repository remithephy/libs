import preprocessing as pp
import file_tool as ft
import numpy as np



file_path = r'C:\Users\0\Desktop\1121.csv'

wavelength,spec = ft.Load_MultiSpec(file_path)
for spec_num in range(len(spec[0])):
    pp.Anomalous_spectrum_removal(spec[:,spec_num])
    spec[:,spec_num] = pp.Normalization(spec[:,spec_num],method='total_intensity')

wavelength = wavelength[:,np.newaxis]


np.savetxt('C:\\Users\\0\\Desktop\\11221.csv',np.concatenate((wavelength,spec),axis = 1),delimiter=',',fmt='%.04f')