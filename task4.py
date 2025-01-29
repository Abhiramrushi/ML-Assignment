import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Polynomial Features from sklearn.linear_model import LinearRegression
file path "plot-data (1) (1).csv"
data pd.read_csv(file_path)
data.columns data.columns.str.strip()
data['x'] pd.to_numeric(data['x'], errors='coerce')
data['y'] = pd.to_numeric(data['y'], errors='coerce')
data data.dropna()
x= data['x'].values.reshape(-1, 1)
ydata['y'].values
degree = 4
poly Polynomial Features(degree-degree)
x_polypoly.fit_transform(X)
model Linear Regression()
model.fit(x_poly, y)
y_pred model.predict(x_poly)
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, y_pred, color='red', label f'Polynomial Regression (degree-(degree)))
plt.title("Polynomial Regression")
plt.xlabel("Tractor Age")
plt.ylabel("Maintainance Cost")
plt.legend()
plt.show()