'''
:@Author: Remi
:@Date: 2023/10/26 10:36:41
:@LastEditors: Remi
:@LastEditTime: 2023/10/26 10:36:41
:Description: 
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR


data = pd.read_csv(r"D:\20231017\preprocessed_for_multi\PLS\weighted_pls.csv")
'''
第一列lable，第二列开始特征，不读第一行
'''
Y = data.values[:, 0]
X = data.values[:, 39:41]

#wl = np.arange(0, len(X[0,:]), 1)
#print(len(wl))
#with plt.style.context('ggplot's):
#    plt.plot(wl, X.T)
#    plt.xlabel("Pixels")
#    plt.ylabel("Absorbance")
#plt.show()
X = X/X.max()
y = Y/1000
# Create and fit the SVR model
svr = SVR(kernel='poly',degree = 7, C=1, epsilon=0.1,gamma = 1.2)
svr.fit(X, y)

# Predict
y_pred = svr.predict(X)
print(y_pred)
# Plot the data points and regression line
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, y_pred, color='red', label='Regression Line')

# Plot the upper and lower boundaries
plt.plot(X, y_pred + svr.epsilon, color='green', linestyle='--', label='Upper Boundary')
plt.plot(X, y_pred - svr.epsilon, color='green', linestyle='--', label='Lower Boundary')

# Calculate R2 score
r2 = r2_score(y, y_pred)

# Add labels and title
plt.xlabel('intensity')
plt.ylabel('concentration(ppm/1000)')
plt.title('SVR Regression')
plt.text(0.95, 0.05, f'R2 score: {r2:.4f}', transform=plt.gca().transAxes, ha='right')
# Add legend
plt.legend()

# Show the plot
plt.show()