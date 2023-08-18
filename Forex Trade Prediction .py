#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install yfinance')
get_ipython().system('pip install pandas_ta')


# In[3]:


import numpy as np
import pandas as pd


# In[4]:


import pandas_ta as pd


# In[5]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# In[6]:


import yfinance as yf


# In[7]:


from scipy.stats import linregress


# In[8]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit


# In[9]:


import joblib


# In[10]:


import warnings
warnings.filterwarnings("ignore")


# In[11]:


from IPython.display import clear_output
clear_output()


# In[12]:


df = yf.download(tickers='EURUSD=X',
                 period='3000d',
                 interval='1d')
df = df.drop(['Volume', 'Adj Close'], axis=1)


# In[13]:


df.head()


# In[14]:


df['ATR'] = df.ta.atr(length=20)
df['RSI'] = df.ta.rsi()
df['Average'] = df.ta.midprice(length=1)
df['MA40'] = df.ta.sma(length=40)
df['MA80'] = df.ta.sma(length=80)
df['MA160'] = df.ta.sma(length=160)


# In[15]:


plt.figure(figsize=(14, 4))
plt.plot(df.index, df.Open)
plt.plot(df.index, df.MA40, ls=':')
plt.plot(df.index, df.MA80, ls=':')
plt.plot(df.index, df.MA160, ls=':')
plt.legend(['Open', 'MA40', 'MA80', 'MA160'])
plt.title('[EUR/USD] Open Price and Moving Averages')
plt.xlabel('Date')
plt.ylabel('EUR/USD')
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.show()


# In[16]:


plt.figure(figsize=(14, 4))
plt.plot(df.index, df.RSI, lw=1, color='gray')
plt.axhline(70, ls='--', color='red')
plt.axhline(30, ls='--', color='red')
plt.legend(['RSI'])
plt.title('Relative Strength Index (RSI)')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.show()


# In[17]:


def get_slope(array):
    y = np.array(array)
    x = np.arange(len(y))
    slope = linregress(x, y)[0]
    return slope


# In[18]:


back_rolling_n = 6
df['slopeMA40'] = df['MA40'].rolling(window=back_rolling_n).apply(get_slope, raw=True)
df['slopeMA80'] = df['MA80'].rolling(window=back_rolling_n).apply(get_slope, raw=True)
df['slopeMA160'] = df['MA160'].rolling(window=back_rolling_n).apply(get_slope, raw=True)
df['AverageSlope'] = df['Average'].rolling(window=back_rolling_n).apply(get_slope, raw=True)
df['RSISlope'] = df['RSI'].rolling(window=back_rolling_n).apply(get_slope, raw=True)


# In[19]:


plt.figure(figsize=(14, 4))
plt.plot(df.index, df.slopeMA40, lw=1, ls=':', color='orange')
plt.plot(df.index, df.slopeMA80, lw=1, ls=':', color='green')
plt.plot(df.index, df.slopeMA160, lw=1, ls=':', color='red')
plt.axhline(0, ls='-.', color='black')
plt.legend(['slopeMA40', 'slopeMA80', 'slopeMA160'])
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.title('Moving Average Slopes')
plt.xlabel('Year')
plt.ylabel('Slope')
plt.show()


# In[20]:


TP_pipdiff = 500e-5
TPSLRatio = 2
SL_pipdiff = TP_pipdiff / TPSLRatio

def get_target(barsupfront, df): 
    length = len(df)
    high = list(df['High'])
    low = list(df['Low'])
    close = list(df['Close'])
    open = list(df['Open'])
    trendcat = [None] * length
    
    for line in range(0, length - barsupfront - 2):
        valueOpenLow = 0
        valueOpenHigh = 0
        for i in range(0, barsupfront + 2):
            value1 = open[line + 1] - low[line + i]
            value2 = open[line + 1] - high[line + i]
            valueOpenLow = max(value1, valueOpenLow)
            valueOpenHigh = min(value2, valueOpenHigh)
            
            if ((valueOpenLow >= TP_pipdiff) and (-valueOpenHigh <= SL_pipdiff)):
                trendcat[line] = 1 # Downtrend
                break
            elif ((valueOpenLow <= SL_pipdiff) and (-valueOpenHigh >= TP_pipdiff)):
                trendcat[line] = 2 # Uptrend
                break
            else:
                trendcat[line] = 0 # No clear trend
                
    return trendcat

df['target'] = get_target(16, df)


# In[21]:


fig = plt.figure(figsize=(20, 8))
ax = fig.gca()
df_model = df[['ATR', 'RSI', 'Average', 'MA40', 'MA80', 'MA160',
              'slopeMA40', 'slopeMA80', 'slopeMA160',
              'AverageSlope', 'RSISlope', 'target']] 
df_model.hist(ax=ax)
plt.suptitle("Histograms for the different features (and the target)")
plt.tight_layout()
plt.show()


# In[22]:


plt.figure(figsize=(6, 3))
df_up = df.RSI[df.target == 2]
df_down = df.RSI[df.target == 1]
df_unclear = df.RSI[df.target == 0]
plt.hist(df_unclear, bins=50, alpha=0.5, label='unclear')
plt.hist(df_down, bins=50, alpha=0.5, label='down')
plt.hist(df_up, bins=50, alpha=0.5, label='up')
plt.title('Distribution of RSI within Target Classes')
plt.xlabel('RSI')
plt.ylabel('Amount')
plt.legend()
plt.show()


# In[23]:


df_model = df_model.dropna()
features = ['ATR', 'RSI', 'Average', 'MA40', 'MA80', 'MA160',
            'slopeMA40', 'slopeMA80', 'slopeMA160', 'AverageSlope', 'RSISlope']
X = df_model[features]
y = df_model['target']
X.head()


# In[24]:


split_index = int(0.8 * len(df_model))
X_train, X_valid = X[:split_index], X[split_index:]
y_train, y_valid = y[:split_index], y[split_index:]


# In[25]:


knn_params = {'n_neighbors': 225,
              'weights': 'uniform',
              'algorithm': 'ball_tree',
              'leaf_size': 30,
              'p': 1,
              'metric': 'minkowski',
              'metric_params': None,
              'n_jobs': 1}

model = KNeighborsClassifier(**knn_params)
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_valid = model.predict(X_valid)
gambler_pred = np.random.choice([0, 1, 2], len(y_pred_valid))


# In[26]:


train_accuracy = accuracy_score(y_train, y_pred_train)
valid_accuracy = accuracy_score(y_valid, y_pred_valid)
gambler_accuracy = accuracy_score(y_valid, gambler_pred)
base_accuracy = max(df_model.target.value_counts().sort_index() / len(df_model))
print(f"Train accuracy:   {train_accuracy * 100 :.2f}%")
print(f"Valid accuracy:   {valid_accuracy * 100 :.2f}%")
print(f"Gambler accuracy: {gambler_accuracy * 100 :.2f}%")
print("\nRepartition of the classes:")
print(df_model.target.value_counts().sort_index() / len(df_model))
print(f"\n=== Accuracy improvement: {(valid_accuracy - base_accuracy) * 100 :.2f}% ===")


# In[ ]:




