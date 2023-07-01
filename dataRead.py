#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from numpy.polynomial import Polynomial
from sklearn.preprocessing import RobustScaler, StandardScaler
from datetime import datetime
import math
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, LinearRegression
from sklearn.tree import DecisionTreeRegressor


from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold, RepeatedKFold, learning_curve
from sklearn.metrics import accuracy_score, f1_score, r2_score, precision_score, recall_score, classification_report, confusion_matrix


# In[2]:


df = pd.read_csv('../tista/Downloads/realestate/real/Real estate1.csv')

df.head()


# In[3]:


df.drop('No', inplace=True, axis=1)


df.columns = ['transaction date', 'house age', 'distance to the nearest MRT station', 'number of convenience stores', 'latitude', 'longitude', 'house price of unit area']
df.head()


# In[4]:


train_data, test_data = train_test_split(df, test_size=0.3, random_state=2)


# In[5]:


print("Train set size:",len(train_data))
print("Test set size:",len(test_data))


# In[6]:


train_data.head()


# In[7]:


train_data.info()


# In[8]:


train_data.describe()


# In[9]:


fig, ax = plt.subplots(2, 3, figsize=(14, 10))
ax = ax.flatten()

sns.set()
sns.lineplot(data=train_data, x="transaction date", y="house price of unit area", ax=ax[0])
ax[0].set_title("Price of Unit Area vs. Transaction Date")

sns.lineplot(data=train_data, x="house age", y="house price of unit area", ax=ax[1])
ax[1].set_title("Price of Unit Area vs. House Age")

sns.lineplot(data=train_data, x="distance to the nearest MRT station", y="house price of unit area", ax=ax[2])
ax[2].set_title("Price of Unit Area vs. Distance to the nearest MRT station")

sns.lineplot(data=train_data, x="number of convenience stores", y="house price of unit area", ax=ax[3])
ax[3].set_title("Price of Unit Area vs. no. of Convenience Stores")

sns.lineplot(data=train_data, x="latitude", y="house price of unit area", ax=ax[4])
ax[4].set_title("Price of Unit Area vs. Latitude")

sns.lineplot(data=train_data, x="longitude", y="house price of unit area", ax=ax[5])
ax[5].set_title("Price of Unit Area vs. Longitude")

# set the spacing between subplots
plt.subplots_adjust(left=0.1,
                    bottom=0, 
                    right=0.9, 
                    top=1, 
                    wspace=0.4, 
                    hspace=0.4)
plt.show();


# In[10]:


#Preparation


# In[11]:


fig = plt.figure(figsize=(26,26))
for index,col in enumerate(train_data):
    plt.subplot(6,3,index+1)
    sns.histplot(train_data.loc[:,col].dropna(), kde=True, stat="density", linewidth=0.5);
fig.tight_layout(pad=1.0);


# In[12]:


fig = plt.figure(figsize=(14,15))
for index,col in enumerate(train_data):
    plt.subplot(6,3,index+1)
    sns.boxplot(y=col, data=train_data.dropna())
    plt.grid()
fig.tight_layout(pad=1.0)


# In[13]:


train_data = train_data[train_data['house price of unit area']<80]
train_data = train_data[train_data['distance to the nearest MRT station']<3000]
train_data = train_data[train_data['longitude']>121.50]


# In[14]:


fig = plt.figure(figsize=(26,26))
for index,col in enumerate(train_data):
    plt.subplot(6,3,index+1)
    sns.histplot(train_data.loc[:,col].dropna(), kde=True, stat="density", linewidth=0.5);
fig.tight_layout(pad=1.0);


# In[15]:


numeric_train = train_data
correlation = numeric_train.corr()
correlation[['house price of unit area']].sort_values(['house price of unit area'], ascending=False)


# In[16]:


corr = train_data.corr()
sns.heatmap(corr, cmap = 'YlGnBu', annot= False, linewidths=.5);


# In[17]:


train_data.head()


# In[18]:


dfs = [train_data, test_data]
data = pd.concat(dfs)


# In[19]:


from yellowbrick.datasets import load_concrete
from yellowbrick.regressor import ResidualsPlot


# In[20]:


X=data.drop('house price of unit area',axis=1)
y=data['house price of unit area']


# In[21]:


transformer = StandardScaler().fit(X)
X_prep = transformer.transform(X)


# In[22]:



polynomial_converter = PolynomialFeatures(degree=2, include_bias=False)


poly_features = polynomial_converter.fit(X_prep)
poly_features = polynomial_converter.transform(X_prep)

poly_features.shape


# In[23]:


#split of train and test dataset


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)


# In[47]:


model_poly = LinearRegression()
model_poly.fit(X_train, y_train)


# In[48]:


pred_train_poly = model_poly.predict(X_train)

r2_train_poly = r2_score(y_train, pred_train_poly)
mse_train_poly = mean_squared_error(y_train, pred_train_poly)
rmse_train_poly = np.sqrt(mse_train_poly)
mae_train_poly = mean_absolute_error(y_train, pred_train_poly)


# In[49]:


pred_val_poly = model_poly.predict(X_val)

r2_val_poly = r2_score(y_val, pred_val_poly)
mse_val_poly = mean_squared_error(y_val, pred_val_poly)
rmse_val_poly = np.sqrt(mse_val_poly)
mae_val_poly = mean_absolute_error(y_val, pred_val_poly)


# In[50]:


pd.DataFrame({'Validation':  [r2_val_poly, mse_val_poly, rmse_val_poly, mae_val_poly],
               'Training': [r2_train_poly, mse_train_poly, rmse_train_poly, mae_train_poly],
             },
              index=['R2', 'MSE', 'RMSE', 'MAE'])


# In[51]:


pred_test_poly = model_poly.predict(X_test)

r2_test_poly = r2_score(y_test, pred_test_poly)
mse_test_poly = mean_squared_error(y_test, pred_test_poly)
rmse_test_poly = np.sqrt(mse_test_poly)
mae_test_poly = mean_absolute_error(y_test, pred_test_poly)

print('R2 Score: ', r2_test_poly)
print('MSE: ', mse_test_poly)
print('RMSE: ', rmse_test_poly)
print('MAE: ', mae_test_poly)


# In[52]:


mean_poly=np.mean(pred_test_poly)
print(mean_poly)


# In[53]:


std_poly=np.std(pred_test_poly)
print(std_poly)


# In[54]:


visualizer = ResidualsPlot(model_poly, hist=False, qqplot=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show();


# In[55]:


pd.DataFrame({'Y_Test': y_test,'Y_Pred':pred_test_poly, 'Residuals':(y_test-pred_test_poly) }).head(5)


# In[56]:


plt.scatter(y_test, pred_test_poly)
plt.xlabel('Real')
plt.ylabel('Pred')
plt.title('Polynomial Reg pred against real')
plt.show()


# In[57]:


model = DecisionTreeRegressor()
model.fit(X_train, y_train)


# In[58]:


pred_train = model.predict(X_train)

r2_train = r2_score(y_train, pred_train)
mse_train = mean_squared_error(y_train, pred_train)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_train, pred_train)


# In[59]:


pred_val = model.predict(X_val)

r2_val = r2_score(y_val, pred_val)
mse_val = mean_squared_error(y_val, pred_val)
rmse_val = np.sqrt(mse_val)
mae_val = mean_absolute_error(y_val, pred_val)


# In[60]:


pd.DataFrame({'Validation':  [r2_val, mse_val, rmse_val, mae_val],
               'Training': [r2_train, mse_train, rmse_train, mae_train],
             },
              index=['R2', 'MSE', 'RMSE', 'MAE'])


# In[61]:


pred_test = model.predict(X_test)

r2_test = r2_score(y_test, pred_test)
mse_test = mean_squared_error(y_test, pred_test)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, pred_test)

print('R2 Score: ', r2_test)
print('MSE: ', mse_test)
print('RMSE: ', rmse_test)
print('MAE: ', mae_test)


# In[40]:


mean_test=np.mean(pred_test)
print(mean_test)


# In[41]:


std_test=np.std(pred_test)
print(std_test)


# In[42]:


visualizer = ResidualsPlot(model, hist=False, qqplot=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show();


# In[43]:


pd.DataFrame({'Y_Test': y_test,'Y_Pred':pred_test, 'Residuals':(y_test-pred_test) }).head(5)


# In[44]:


plt.scatter(y_test, pred_test)
plt.xlabel('Real')
plt.ylabel('Pred')
plt.title('Polynomial Reg pred against real')
plt.show()


# In[45]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# In[46]:


forest_model = RandomForestRegressor(random_state=3)
forest_model.fit(X_train, y_train)


# In[54]:


predi_train = forest_model.predict(X_train)

r2_train_for = r2_score(y_train, predi_train)
mse_train_for = mean_squared_error(y_train, predi_train)
rmse_train_for = np.sqrt(mse_train_for)
mae_train_for = mean_absolute_error(y_train, predi_train)


# In[55]:


pred_val_for = forest_model.predict(X_val)
r2_val_for = r2_score(y_val, pred_val_for)
mse_val_for = mean_squared_error(y_val, pred_val_for)
rmse_val_for = np.sqrt(mse_val)
mae_val_for = mean_absolute_error(y_val, pred_val_for)


# In[56]:


pd.DataFrame({'Validation':  [r2_val_for, mse_val_for, rmse_val_for, mae_val_for],
               'Training': [r2_train_for, mse_train_for, rmse_train_for, mae_train_for],
             },
              index=['R2', 'MSE', 'RMSE', 'MAE'])


# In[57]:


pred_test_for = forest_model.predict(X_test)

r2_test_for = r2_score(y_test, pred_test_for)
mse_test_for = mean_squared_error(y_test, pred_test_for)
rmse_test_for = np.sqrt(mse_test)
mae_test_for = mean_absolute_error(y_test, pred_test_for)

print('R2 Score: ', r2_test_for)
print('MSE: ', mse_test_for)
print('RMSE: ', rmse_test_for)
print('MAE: ', mae_test_for)


# In[58]:


mean_for=np.mean(pred_test_for)
print(mean_for)


# In[59]:


std_for=np.std(pred_test_for)
print(std_for)


# In[61]:


visualizer = ResidualsPlot(forest_model, hist=False, qqplot=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show();


# In[62]:


pd.DataFrame({'Y_Test': y_test,'Y_Pred':pred_test_for, 'Residuals':(y_test-pred_test_for) }).head(5)


# In[63]:


plt.scatter(y_test, pred_test_for)
plt.xlabel('Real')
plt.ylabel('Pred')
plt.title('Polynomial Reg pred against real')
plt.show()


# In[64]:


from xgboost import XGBRegressor


# In[65]:


my_model = XGBRegressor()
my_model.fit(X_train, y_train)


# In[66]:


predic_train = my_model.predict(X_train)

r2_train_xgb = r2_score(y_train, predic_train)
mse_train_xgb = mean_squared_error(y_train, predic_train)
rmse_train_xgb = np.sqrt(mse_train_xgb)
mae_train_xgb = mean_absolute_error(y_train, predic_train)


# In[67]:


predic_val_xgb = my_model.predict(X_val)

r2_val_xgb = r2_score(y_val, predic_val_xgb)
mse_val_xgb = mean_squared_error(y_val, predic_val_xgb)
rmse_val_xgb = np.sqrt(mse_val_xgb)
mae_val_xgb = mean_absolute_error(y_val, predic_val_xgb)


# In[68]:


pd.DataFrame({'Validation':  [r2_val_xgb, mse_val_xgb, rmse_val_xgb, mae_val_xgb],
               'Training': [r2_train_xgb, mse_train_xgb, rmse_train_xgb, mae_train_xgb],
             },
              index=['R2', 'MSE', 'RMSE', 'MAE'])


# In[69]:


pred_test_xgb = my_model.predict(X_test)

r2_test_xgb = r2_score(y_test, pred_test_xgb)
mse_test_xgb = mean_squared_error(y_test, pred_test_xgb)
rmse_test_xgb = np.sqrt(mse_test)
mae_test_xgb = mean_absolute_error(y_test, pred_test_xgb)

print('R2 Score: ', r2_test_xgb)
print('MSE: ', mse_test_xgb)
print('RMSE: ', rmse_test_xgb)
print('MAE: ', mae_test_xgb)


# In[70]:


mean_xgb=np.mean(pred_test_xgb)
print(mean_xgb)


# In[75]:


std_xgb=np.std(pred_test_xgb)
print(std_xgb)


# In[72]:


visualizer = ResidualsPlot(my_model, hist=False, qqplot=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show();


# In[73]:


pd.DataFrame({'Y_Test': y_test,'Y_Pred':pred_test_xgb, 'Residuals':(y_test-pred_test_xgb) }).head(5)


# In[74]:


plt.scatter(y_test, pred_test_xgb)
plt.xlabel('Real')
plt.ylabel('Pred')
plt.title('Polynomial Reg pred against real')
plt.show()


# In[ ]:





# In[ ]:




