import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def compare(sample, subplotNum1, subplotNum3, subplotNum9):
    x1 = np.linspace(0, 2*np.pi, sample)
    y1 = np.sin(x1)+np.random.randn(len(x1))/5.0
    plt.subplot(3, 3, subplotNum1)
    plt.scatter(x1, y1)
    x1 = x1.reshape(-1, 1)
    slr = LinearRegression()
    slr.fit(x1, y1)
    print("1-polynomial")
    print("Regression coefficients:", slr.coef_)
    print("intercept:", slr.intercept_)
    predicted_y1 = slr.predict(x1)
    plt.subplot(3, 3, subplotNum1)
    plt.plot(x1, predicted_y1)
    # 線圖

    poly_features_3 = PolynomialFeatures(degree=3, include_bias=False)
    X_poly_3 = poly_features_3.fit_transform(x1)
    lin_reg_3 = LinearRegression()
    lin_reg_3.fit(X_poly_3, y1)
    print("3-polynomial")
    print("Regression coefficients:", lin_reg_3.coef_)
    print("intercept:", lin_reg_3.intercept_, lin_reg_3.coef_)
    X_plot = np.linspace(0, 6, 1000).reshape(-1, 1)
    X_plot_poly = poly_features_3.fit_transform(X_plot)
    y_plot = np.dot(X_plot_poly, lin_reg_3.coef_.T) + lin_reg_3.intercept_
    plt.subplot(3, 3, subplotNum3)
    plt.plot(X_plot, y_plot, 'r-')
    plt.plot(x1, y1, 'b.')

    poly_features_d = PolynomialFeatures(degree=9, include_bias=False)
    X_poly_d = poly_features_d.fit_transform(x1)
    lin_reg_d = LinearRegression()
    lin_reg_d.fit(X_poly_d, y1)
    print("9-polynomial")
    print("Regression coefficients:", lin_reg_d.coef_)
    print("intercept:", lin_reg_d.intercept_)
    X_plot = np.linspace(0, 6, 1000).reshape(-1, 1)
    X_plot_poly = poly_features_d.fit_transform(X_plot)
    y_plot = np.dot(X_plot_poly, lin_reg_d.coef_.T) + lin_reg_d.intercept_
    plt.subplot(3, 3, subplotNum9)
    plt.plot(X_plot, y_plot, 'r-')
    plt.plot(x1, y1, 'b.')
compare(10, 1, 4, 7)
compare(50, 2, 5, 8)
compare(100, 3, 6, 9)
plt.show()
