import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import preprocessing as pp
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression


data = pd.read_csv(r"D:\20231017\preprocessed_for_multi\PLS\normean_pls.csv")


Y = data.values[:, 0]
X = data.values[:, 1:]

######################## Plot the data
wl = np.arange(0, len(X[0,:]), 1)
print(len(wl))
with plt.style.context('ggplot'):
    plt.plot(wl, X.T)
    plt.xlabel("Pixels")
    plt.ylabel("Absorbance")

# X2 =  - savgol_filter(X, 3, polyorder=2, deriv=2)
# ####################### plot savgol_filter data
# plt.figure(figsize=(8, 4.5))
# with plt.style.context('ggplot'):
#     plt.plot(wl, X2.T)
#     plt.xlabel("Pixels")
#     plt.ylabel("After Savitzky-Golay Filter")
#     plt.show()

################## test with n-2 components
r2s = []
#r2s2 = []
mses = []
xticks = np.arange(1, 5)
for n_comp in xticks:
    Ypredict, r2, mse = pp.optimise_pls_cv(X, Y, n_comp)
    #Y2, r22, mse2 = pp.optimise_pls_cv(X2, Y, n_comp)
    r2s.append(r2)
    #r2s2.append(r22)
    mses.append(mse)

#################### Draw mse ,r2 and (vip for max r2)
MSE_score = pp.plot_metrics(xticks,mses, 'MSE', 'min')
R2_score = pp.plot_metrics(xticks,r2s, 'R2', 'max')
#R22_score = pp.plot_metrics(xticks,r2s2, 'R2', 'max')
pls = PLSRegression(n_components = R2_score)
pls.fit(X,Y)
vips = pp._calculate_vips(pls)
with plt.style.context('ggplot'):
    plt.plot(wl, vips, '-r', label='vips')
    plt.show()


################### F" data
Ypredict, r21, mse = pp.optimise_pls_cv(X, Y, R2_score)
#Y2, r22, mse= pp.optimise_pls_cv(X2, Y, R22_score)

################### plot all
plt.figure(figsize=(6, 6))
with plt.style.context('ggplot'):
    plt.scatter(Y, Ypredict, color='red')
    #plt.scatter(Y, Y2, color='blue')
    
    z1 = np.polyfit(Y, Ypredict, 1)
    #z2 = np.polyfit(Y, Y2, 1)

    plt.plot(Y, Y, '-g', label='Expected regression line')
    plt.plot( Y, np.polyval(z1, Y),color='r', label=('Predicted Y ,R2 = ' + str(r21)))
    #plt.plot( Y, np.polyval(z2, Y),color='blue', label=('Predicted Y" ,R2 = ' + str(r22)))   
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.plot()
plt.show()