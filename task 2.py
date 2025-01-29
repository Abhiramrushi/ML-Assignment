import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
df = pd.read_csv('Problem 4.csv')
print(df.head())
print(df.isna().sum())
sns.lmplot(data=df, x='TV', y='Sales', ci=None, line_kws={'color': 'blue'})
plt.title('Sales vs TV Advertising Spend')
plt.show()
sns.lmplot(data=df, x='Radio', y='Sales', ci=None, line_kws={'color': 'green'})
plt.title('Sales vs Radio Advertising Spend')
plt.show()
sns.lmplot(data=df, x='Newspaper', y='Sales', ci=None, line_kws={'color': 'red'})
plt.title('Sales vs Newspaper Advertising Spend')
plt.show()
cor_matrix = df[['TV', 'Radio', 'Newspaper', 'Sales']].corr()
print(cor_matrix)
X_tv = df[['TV']]
y = df['Sales']
model_tv = sm.OLS(y, sm.add_constant(X_tv)).fit()
print(model_tv.summary())
def predict_sales(tv_spend):
    new_data = pd.DataFrame({'TV': [tv_spend]})
    new_data = sm.add_constant(new_data)
    predicted_sales = model_tv.predict(new_data)
    return predicted_sales.iloc[0]
tv_ad_spend = 150  
predicted_sales = predict_sales(tv_ad_spend)
print(f'Predicted Sales for TV advertising spend of {tv_ad_spend}: {predicted_sales}')
X_multiple = df[['TV', 'Radio', 'Newspaper']]
model_multiple = sm.OLS(y, sm.add_constant(X_multiple)).fit()
print(model_multiple.summary())
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(df[['TV', 'Radio']])
X_poly = np.concatenate([X_poly, df[['Newspaper']].values], axis=1)
model_poly = sm.OLS(y, sm.add_constant(X_poly)).fit()
print(model_poly.summary())
def predict_sales_poly(tv_spend, radio_spend, newspaper_spend):
    transformed = poly.transform([[tv_spend, radio_spend]])
    new_data = np.concatenate([transformed, [[newspaper_spend]]], axis=1)
    new_data = sm.add_constant(new_data)
    predicted_sales = model_poly.predict(new_data)
    return predicted_sales[0]
tv_ad_spend2 = 150
radio_ad_spend2 = 50
newspaper_ad_spend2 = 30
predicted_sales_poly = predict_sales_poly(tv_ad_spend2, radio_ad_spend2, newspaper_ad_spend2)
print(f'Predicted Sales for TV spend {tv_ad_spend2}, Radio spend {radio_ad_spend2}, Newspaper spend {newspaper_ad_spend2}: {predicted_sales_poly}')
interaction_term = df['TV'] * df['Radio']
X_interaction = df[['TV', 'Radio', 'Newspaper']].copy()
X_interaction['TV_Radio'] = interaction_term
model_interaction = sm.OLS(y, sm.add_constant(X_interaction)).fit()
print(model_interaction.summary())