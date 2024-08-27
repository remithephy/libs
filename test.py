import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 定义Lorentzian函数
def lorentzian(x, x0, A, gamma):
    return A * gamma**2 / ((x - x0)**2 + gamma**2)

# 定义双Lorentzian函数作为拟合模型
def double_lorentzian(x, x1, A1, gamma1, x2, A2, gamma2):
    return lorentzian(x, x1, A1, gamma1) + lorentzian(x, x2, A2, gamma2)

# 生成模拟的光谱数据
x_data = np.linspace(840, 860, 200)
true_params = [852, 1024, 2, 857, 819, 3]  
y_data = double_lorentzian(x_data, *true_params) + np.random.normal(0, 20, len(x_data))  

wavelength = x_data
intensity = y_data

# 进行拟合
initial_guess = [600, 1000, 20, 650, 800, 30]  
fit_params, _ = curve_fit(double_lorentzian, wavelength, intensity, p0=initial_guess,maxfev = 50000)

# 绘制拟合结果
plt.plot(wavelength, intensity, label='Original Spectrum')
fit_curve = double_lorentzian(wavelength, *fit_params)
plt.plot(wavelength, fit_curve, color='red', label='Double Lorentz Fit')

# 标注峰的位置
for i in range(0, len(fit_params), 3):
    peak_position = fit_params[i]
    peak_intensity = lorentzian(peak_position, *fit_params[i:i+3])
    
    plt.plot(peak_position, peak_intensity, 'bo')  # 标注峰的位置
    
    # 绘制垂直虚线
    plt.plot([peak_position, peak_position], [0, peak_intensity], color='gray', linestyle='--')

    # 添加标签
    #plt.text(peak_position, peak_intensity, f'({peak_position:.1f}, {peak_intensity:.1f})', fontsize=9, color='blue')


plt.plot(wavelength, lorentzian(wavelength, fit_params[0], fit_params[1], fit_params[2]),
         linestyle='dashed', color='green', label='Lorentz 1: ')
plt.plot(wavelength, lorentzian(wavelength, fit_params[3], fit_params[4], fit_params[5]),
         linestyle='dashed', color='blue', label='Lorentz 2: ')

plt.legend()
plt.xlabel('Wavelength')
plt.ylabel('Intensity')
plt.title('Double Lorentz Peak Fitting')

plt.show()

print('Fit parameters:', fit_params)

