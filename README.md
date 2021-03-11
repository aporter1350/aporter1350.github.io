# CATS
Welcome to our blog for CATS (Constructing Algorithmic Trading Structures) — a market data project completed for STAT 359 in winter 2021. The goal of our project is to construct a data processing pipeline that can retreive, process, and analyze level 1 and level 2 market data.

## Pipeline Overview
The framework of our pipeline is described by the following:
<img width="726" alt="Screen Shot 2021-03-11 at 10 14 55 AM" src="https://user-images.githubusercontent.com/78179650/110818712-220b6280-8253-11eb-9587-64cbc6e78f34.png">  
In the following sections we will discuss each part of the pipeline in further detail.

## Acquiring level 1 and level 2 data
People have been trying to predict the stock market for centuries. Many investment strategies involve using different types of market data to predict future stock behavior. With advancements in acquiring real time market book data, investors can now apply many different types of analyses to identify new trading opportunities that could be profitable for investors. But this comes at a cost, streaming data in real time requires a thoroughly vetted pipeline that can process and store data in a way that is flexible for use. This often requires customized hardware and a dedicated location for storing incoming streams of data.

The first part of our pipeline involved collecting level 1 and level 2 data. Data allocation was done using a cloud [platform](https://iexcloud.io/blog/how-to-get-market-data-in-python). IEX cloud is a platform that provides financial data to clients in order to use IEX cloud in python users must download [pyEx](https://github.com/timkpaine/pyEX) and create an account on IEX cloud in order to receive an API key for streaming. Level 1 data is fairly easy to acquire and can be done in python.

```
import pyEX as p
c = p.Client(api_token=YOUR API KEY)
sym='IEX' #stock symbol
timeframe='5y' #timeframe
df = c.chartDF(symbol=sym, timeframe=timeframe)
df.to_csv("~/.qlib/csv_data/5Y_IEX.csv")
```

To get the data in a format for qlib, open the terminal and type

```
python scripts/dump_bin.py dump_all --csv_path ~/.qlib/csv_data/IEX.csv --qlib_dir ~/.qlib/qlib_data/iex --include_fields close,high,low,open,volume --date_field_name date
```

Acquiring level 2 data often involves one of two methods: the first would be to pay a premium to vendors who calculate, organize, and collect ongoing data for you. The alternative method is to stream data for yourself, giving the user more control on incoming data streams. This involves the use of whats called a server sent event. This is a process that enables a user to receive automatic updates from the IEX website. What this means is that a user can create a link to a website using an API key. The script would then check on the website to see if a particular stock has changed. If it has the website will then send information back to the user usually in the form of a dictionary like object consisting of bid price, bid size, ask price, ask size, and the time. And this process can occur indefinitely but keep in mind that most users do not have the storage capacity to run this script forever, it also costs money to receive ongoing level 2 data. So unless you’re paying for a premium subscription, you’re fairly limited in how much data you can retrieve. Streaming can sometimes be considered more efficient because you are receiving the latest available data. Endpoints can be adjusted to specific intervals such as 1second, 5 seconds, or 1 minute.

```
tmp=[]
symbol='iex' #stock symbol
channels='book' #market book data
c.deepSSE(symbol,'book',on_data=tmp.append)
```

## Data Processing
One of our main tools for processing data is Qlib, an AI-oriented investment platform designed by Microsoft. Extensive documentation for Qlib can be found [here](https://qlib.readthedocs.io/en/latest/), and the official repository can be found [here](https://github.com/microsoft/qlib). Qlib contains an extensive framework of modules that all serve different purposes in market data processing. However, these modules are designed to be standalone pieces of code, which in turn allows us to focus on the ones that are most relevant to our project.

### Dumping data into Qlib
Extensive information on how to input, or "dump" data into Qlib can be found in the above documentation and is also described above. However, in some cases  preprocessing of the data might be required in order to successfully dump data into Qlib. This might be the case when the data contains duplicate values, etc. An example of a dataset that presents this issue (sampleMSFTdata.csv), as well as a preprocessing script to fix this problem (qlibdump.py), can be found in the src folder. This script is also shown below for reference.
```
import pandas as pd
data = pd.read_csv('sampleMSFTdata.csv')
data = data.drop_duplicates(subset='t', keep='first')
data.to_csv("MSFT.csv", index=False)
```

### Qlib Processing Classes
Qlib provides many useful data-processing child classes. An exhaustive list can be found in the aforementioned documentation, but the ones that are most relevant to our project can be found in the processor (processor.py). The processor also contains some custom-built child classes that we created to process our specific datasets. For example, the FormatLevelTwo child class was created to reformat realtime level 2 data obtained from the IEX cloud service. An example of how these child classes can be used to process data is provided in the src file (processorexample.ipynb).

## Feature Engineering
Financial indicators are one way to analyze changes in the stock market. Each indicator is derived slightly differently and can help explain patterns in market level trends, for the purpose of this project we constructed several different financial indicators to be used as features for modeling. We also tried calculating some of these indicators on level 2 data to see if it would be useful in explaining day to day fluctuations in market book data. Often times financial indicators are fit to the closing price from level 1 data. For our project we applied the same calculations to level 2 data using an average of the bid price.

### Exponential Moving Average (EMA)
The exponential moving average is a type of moving average that places a greater weight and significance on the most recent data points. An EMA reacts more to recent price changes than a simple moving average. This is helpful for observing longer periods of time and smooths out random fluctuations in the data. EMA can also be used to calculate other more complex feature indicators such as MACD
```
import pyEX as p
import pandas as pd
import seaborn as sns
from jupyterthemes import jtplot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
jtplot.style(theme='onedork',context='talk',fscale=1.6)
from matplotlib.lines import Line2D
import talib as t #talib provides all kinds of tools for technical indicators
c = p.Client(api_token=YOUR API KEY)
sym='IEX'
col='close'
period=14 #RSI typically takes the first 14 days to generate an average change score
df = c.chartDF(symbol=sym, timeframe='1y')  #need a lot of data for a good plot
df.reset_index(inplace=True)
df.sort_values(by=['date'],inplace=True)
custom_lines = [
                Line2D([0], [0], color='blue', lw=4),
                Line2D([0], [0], color='green', lw=4)]
jtplot.style(theme='onedork',context='talk',fscale=1.6)
df['date']=pd.to_datetime(df['date'])
df['ema']= t.EMA(df['close'].values.astype(float))
ax=sns.lineplot(x='date',y='ema',data=df,color='green')
sns.lineplot(x='date',y='close',ci=None,data=df, alpha=.5,color='orange')
plt.ylabel("Exponential Moving Average", fontsize=15)
plt.ylabel("Close", fontsize=15)
plt.xlabel("Date", fontsize=15)
ax.legend(custom_lines, ['EMA','Close'],loc='upper left',fontsize=15)
```
![](https://raw.githubusercontent.com/aporter1350/aporter1350.github.io/gh-pages/src/images/EMAL1.png)

### Moving Average Convergence Divergence (MACD)
MACD is a trend following momentum indicator and is calculated by subtracting a 26 period EMA from a 12 period EMA. This results in the MACD line. A 9 period EMA is often referred to as the signal line and is usually plotted on top of the MACD line. This functions as a signal to suggest when to buy and sell. When the MACD crosses above the signal line this would indicate to buy and similarly below the signal line indicates when to sell.

```
macd, macdsignal,macdhist=t.MACD(df['close'].values.astype(float))
df['macd']=macd
df['macdsignal']=macdsignal
df['macdhist']=macdhist
fig=plt.figure(figsize=(15,10), constrained_layout=True)
plt.rcParams['figure.constrained_layout.use'] = True
gs = gridspec.GridSpec(nrows=2, ncols=1, wspace=.2, hspace=.3)
ax0=fig.add_subplot(gs[0,0])
ax=sns.lineplot(x='date',y='close',data=df,ax=ax0)
ax.set_title('IEX Price Level 1',fontsize=20)
plt.ylabel("Close", fontsize=15)
plt.xlabel("Date", fontsize=15)
ax1=fig.add_subplot(gs[1,0],sharex=ax0)
ax=sns.lineplot(x='date',y='macd',data=df,ax=ax1,color='orange')
ax=sns.lineplot(x='date',y='macdsignal',data=df,ax=ax1,color='green')
plt.ylabel("MACD", fontsize=15)
plt.xlabel("Date", fontsize=15)
ax.legend(custom_lines, ['MACD','Signal'],loc='upper left',fontsize=15)
```
![](https://raw.githubusercontent.com/aporter1350/aporter1350.github.io/gh-pages/src/images/MACDL1.png)

We derived MACD using level 2 data from the average weight of a bid price. As you can see from the plot using level 2 data can result in more spurious trends in the data making is harder to parse out information compared to using level 1 data.

![](https://raw.githubusercontent.com/aporter1350/aporter1350.github.io/gh-pages/src/images/MACD.png)

### Relative Strength Index (RSI)
Relative strength index is a momentum indicator. It measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock. The RSI plot ranges from 0 to 100 and is meant to indicate that values greater than 70 indicate that a stock is overbought or overvalued and may be primed for a pullback in price. While an RSI below 30 indicates an oversold or undervalued stock. This measure is derived by calculating the initial average gain or loss during the first 14 timepoints. Each time point after that measures the relative gain or loss of a price. The primary point of an RSI plot is to tell you what kinds of trends are to be expected. RSI and MACD are similar in that both measure the momentum of an asset, however they measure different factors and can sometimes give contradictory results. MACD is meant to show you the relationship between two EMA’s while the RSI measures price change in relation to recent price highs and lows. Many analysts often use RSI and MACD together to provide a better technical picture of a market.
```
df['rsi']= t.RSI(df[col].values.astype(float), period) #Calculate RSI Values
plt.figure(figsize=(15,10), constrained_layout=True)
plt.rcParams['figure.constrained_layout.use'] = True
gs = gridspec.GridSpec(nrows=2, ncols=1, wspace=.2, hspace=.3)
ax0=fig.add_subplot(gs[0,0])
ax=sns.lineplot(x='date',y='close',data=df,ax=ax0)
ax.set_title('IEX Price Level 1',fontsize=20)
plt.ylabel("Close", fontsize=15)
plt.xlabel("Date", fontsize=15)
ax1=fig.add_subplot(gs[1,0],sharex=ax0)
ax=sns.lineplot(x='date',y='rsi',data=df,ax=ax1)
ax.set_title('RSI Chart',fontsize=20)
plt.axhline(0, linestyle='--', alpha=0.1)
plt.axhline(20, linestyle='--', alpha=0.5)
plt.axhline(30, linestyle='--')
plt.ylabel("Relative Strength Index", fontsize=15)
plt.xlabel("Date", fontsize=15)
plt.axhline(70, linestyle='--')
plt.axhline(80, linestyle='--', alpha=0.5)
plt.axhline(100, linestyle='--', alpha=0.1)
```
![](https://raw.githubusercontent.com/aporter1350/aporter1350.github.io/gh-pages/src/images/rsi_L1.png)

We can also derive the RSI values for level 2 data using the bid price. From this dataset you can see that at around 2:35pm we see a dip below 30, this would suggest that the stock is being undervalued and that it may be a good time to buy. And we notice on the plot above that shows the changes in bid price after 2:35 does reveal an increase in market price suggesting it would have been a good time to buy that day.


![](https://raw.githubusercontent.com/aporter1350/aporter1350.github.io/gh-pages/src/images/RSI.png)

## Bollinger Band
Bollinger band is one of the more commonly used financial indicators. A bollinger band is calculated by taking two standard deviations away from a simple moving average. The plot is composed of three lines. A simple moving average denoted as the middle line, and an upper and lower band. The standard deviations typically are calculated from a 20 period simple moving average but this number can be modified based on the user. Bollinger bands are typically interpreted based on trend seen in the upper and lower bounds. If the price moves towards the upper bound this would suggest a stock is being over valued and should be sold. If the price moves towards the lower bound suggests the opposite, that the price is being undervalued and should be bought. The bollinger band is meant to show a stocks volatility, when the market is more volatile the bands get larger, less volatile, the bands shrink.

```
up,mid,low= t.BBANDS(df[col].values.astype(float),20)
df['uppr']=up
df['mid']=mid
df['low']=low
custom_lines = [Line2D([0], [0], color='orange', lw=4),
               Line2D([0], [0], color='green', lw=4),
               Line2D([0], [0], color='red', lw=4),]
fig=plt.figure(figsize=(15,10), constrained_layout=True)
plt.rcParams['figure.constrained_layout.use'] = True
gs = gridspec.GridSpec(nrows=2, ncols=1, wspace=.2, hspace=.3)
ax0=fig.add_subplot(gs[0,0])
ax=sns.lineplot(x='date',y='close',data=df,ax=ax0)
ax.set_title('IEX Price Level 1',fontsize=20)
plt.ylabel("Close", fontsize=15)
plt.xlabel("Date", fontsize=15)
ax1=fig.add_subplot(gs[1,0],sharex=ax0)
ax=sns.lineplot(x='date',y='mid',data=df,ax=ax1,color='orange')
sns.lineplot(x='date',y='uppr',data=df,ax=ax1,color='green')
sns.lineplot(x='date',y='low',data=df,ax=ax1,color='red')
ax.set_title('Bollinger Band',fontsize=20)
plt.ylabel("Relative Strength Index", fontsize=15)
plt.xlabel("Date", fontsize=15)
ax.legend(custom_lines, ['20 Day Mean Average','Upper Bound','Lower Bound'],loc='lower right',fontsize=15)
```

![](https://raw.githubusercontent.com/aporter1350/aporter1350.github.io/gh-pages/src/images/BBL1.png)

## Modeling

### Modeling Overview

### Our Models - Level 1 Data
After using Qlib to process our data into an acceptible format, we can use the features of XGBoost modeling to create models. This process of XGBoost modeling can be applied to our level 1 IEX stock data. That is, we can create an XGBoost model that trains on technical indicators (in this case, we use the MACD) to try to predict the price of IEX stock at any given time. Documentation for how XGBoost can be applied to stock market data can be found [here](https://www.kaggle.com/mtszkw/xgboost-for-stock-trend-prices-prediction). The following is an application of this modeling that can also be found in the src folder (XGBoostModelLevel1.ipynb), and the result of the modeling can be found below.
```
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import plotly as py
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from warnings import simplefilter
import pyEX as p
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)
import talib as t
import warnings
warnings.filterwarnings("ignore")
c = p.Client(api_token=YOUR API KEY)
sym='IEX'
timeframe='1y'
df = c.chartDF(symbol=sym, timeframe=timeframe)
init_notebook_mode(connected=True)

layout=go.Layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(103, 128, 159, .8)')
fig = go.Figure(layout=layout)
fig.update_layout(
    font_color="white",
)
templated_fig = pio.to_templated(fig)
pio.templates['my_template'] = templated_fig.layout.template
pio.templates.default = 'my_template'
macd, macdsignal,macdhist=t.MACD(df['close'].values.astype(float))
df['macd']=macd
df['macdsignal']=macdsignal
df = df.iloc[33:] # Remove inital starting point because of moving averages and MACD line
df = df[:-1]      # Because of shifting close price

df = df.iloc[33:] # Because of moving averages and MACD line
df = df[:-1]      # Because of shifting close price

df.index = range(len(df))

test_size  = 0.15
valid_size = 0.15

test_split_idx  = int(df.shape[0] * (1-test_size))
valid_split_idx = int(df.shape[0] * (1-(valid_size+test_size)))

train_df  = df.loc[:valid_split_idx].copy()
valid_df  = df.loc[valid_split_idx+1:test_split_idx].copy()
test_df   = df.loc[test_split_idx+1:].copy()
#Overview of train, test, and validation sets 
fig = go.Figure()
fig.add_trace(go.Scatter(x=train_df.date, y=train_df.close, name='Training'))
fig.add_trace(go.Scatter(x=valid_df.date, y=valid_df.close, name='Validation'))
fig.add_trace(go.Scatter(x=test_df.date,  y=test_df.close,  name='Test'))
fig.show()

#Predict closing price 
drop_cols = ['date', 'volume', 'open', 'low', 'high', 'symbol', 'id', 'key', 'updated', 'label', 'subkey']

train_df = train_df.drop(drop_cols, 1)
valid_df = valid_df.drop(drop_cols, 1)
test_df  = test_df.drop(drop_cols, 1)

y_train = train_df['close'].copy()
X_train = train_df.drop(['close'], 1)

y_valid = valid_df['close'].copy()
X_valid = valid_df.drop(['close'], 1)

y_test  = test_df['close'].copy()
X_test  = test_df.drop(['close'], 1)

#Setup model parameters 
parameters = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.001, 0.005, 0.01, 0.05],
    'max_depth': [8, 10, 12, 15],
    'gamma': [0.001, 0.005, 0.01, 0.02],
    'random_state': [42]
}
eval_set = [(X_train, y_train), (X_valid, y_valid)]
#Initialize model
model = xgb.XGBRegressor(eval_set=eval_set, objective='reg:squarederror', verbose = False)
clf = GridSearchCV(model, parameters)
#Train model using prior year worth of data 
model = xgb.XGBRegressor(eval_set = [(X_train, y_train), (X_valid, y_valid)], objective='reg:squarederror')
model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

#Predict using untrained recent data 
y_pred = model.predict(X_test)
print(f'y_true = {np.array(y_test)[:50]}')
print(f'y_pred = {y_pred[:50]}')

predicted_prices = df.loc[test_split_idx+1:].copy()
predicted_prices['Close'] = y_pred

fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Scatter(x=df.date, y=df.close,
                         name='Truth',
                         marker_color='LightSkyBlue'), row=1, col=1)

fig.add_trace(go.Scatter(x=predicted_prices.date,
                         y=predicted_prices.close,
                         name='Prediction',
                         marker_color='MediumPurple'), row=1, col=1)

fig.add_trace(go.Scatter(x=predicted_prices.date,
                         y=y_test,
                         name='Truth',
                         marker_color='LightSkyBlue',
                         showlegend=False), row=2, col=1)

fig.add_trace(go.Scatter(x=predicted_prices.date,
                         y=y_pred,
                         name='Prediction',
                         marker_color='MediumPurple',
                         showlegend=False), row=2, col=1)

fig.show()

```
![newplot (8)](https://user-images.githubusercontent.com/78179650/110850524-7e7f7980-8275-11eb-9554-0eb35dd8871c.png)  
(If only the chart is visible, try viewing this blog on our [themed webpage](https://aporter1350.github.io/)!)

### Our Models - Level 2 Data
We also created an XGBoost model for our level 2 IEX market data. The process for creating this model was very similar as the above model (XGBoostModelLevel2.ipynb). However, this time we are instead predicting a weighted average of the bid and ask prices, since level 2 data might not contain the actual prices at which a stock was traded at. The code and results of this model can be found below. 
```
import pyEX as p
import pandas as pd
import seaborn as sns
import numpy as np
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import plotly as py
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

init_notebook_mode(connected=True)

#Input data and create a weighted average of bid and ask prices
df = pd.read_csv('processedlevel2data.csv')
df['average'] = df[['bidprice', 'askprice']].mean(axis=1)
df['average'] = np.where(df['average'] < 150, df[['bidprice', 'askprice']].max(axis=1), df['average'])
df['time'] = df['asktimestamp'].copy()
df.time.fillna(df.bidtimestamp, inplace=True)
df = df[df.average != 0]


# Change default background color for all visualizations
layout=go.Layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(103, 128, 159, .8)')
fig = go.Figure(layout=layout)
fig.update_layout(
    font_color="white",
)
templated_fig = pio.to_templated(fig)
pio.templates['my_template'] = templated_fig.layout.template
pio.templates.default = 'my_template'


df = df.iloc[33:] # Because of moving averages and MACD line
df = df[:-1]      # Because of shifting close price

df.index = range(len(df))

test_size  = 0.15
valid_size = 0.15

test_split_idx  = int(df.shape[0] * (1-test_size))
valid_split_idx = int(df.shape[0] * (1-(valid_size+test_size)))

train_df  = df.loc[:valid_split_idx].copy()
valid_df  = df.loc[valid_split_idx+1:test_split_idx].copy()
test_df   = df.loc[test_split_idx+1:].copy()

fig = go.Figure()
fig.add_trace(go.Scatter(x=train_df.time, y=train_df.average, name='Training'))
fig.add_trace(go.Scatter(x=valid_df.time, y=valid_df.average, name='Validation'))
fig.add_trace(go.Scatter(x=test_df.time,  y=test_df.average,  name='Test'))
fig.show()

drop_cols = ['symbol', 'messageType', 'bidtimestamp', 'asktimestamp', 'time']

train_df = train_df.drop(drop_cols, 1)
valid_df = valid_df.drop(drop_cols, 1)
test_df  = test_df.drop(drop_cols, 1)


y_train = train_df['average'].copy()
X_train = train_df.drop(['average'], 1)

y_valid = valid_df['average'].copy()
X_valid = valid_df.drop(['average'], 1)

y_test  = test_df['average'].copy()
X_test  = test_df.drop(['average'], 1)

X_train.info()

parameters = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.001, 0.005, 0.01, 0.05],
    'max_depth': [8, 10, 12, 15],
    'gamma': [0.001, 0.005, 0.01, 0.02],
    'random_state': [42]
}

eval_set = [(X_train, y_train), (X_valid, y_valid)]
model = xgb.XGBRegressor(eval_set=eval_set, objective='reg:squarederror', verbose = False)

model = xgb.XGBRegressor(eval_set = [(X_train, y_train), (X_valid, y_valid)], objective='reg:squarederror')
model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

y_pred = model.predict(X_test)
print(f'y_true = {np.array(y_test)[:50]}')
print(f'y_pred = {y_pred[:50]}')

predicted_prices = df.loc[test_split_idx+1:].copy()
predicted_prices['Close'] = y_pred

fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Scatter(x=df.time, y=df.average,
                         name='Truth',
                         marker_color='LightSkyBlue'), row=1, col=1)

fig.add_trace(go.Scatter(x=predicted_prices.time,
                         y=predicted_prices.average,
                         name='Prediction',
                         marker_color='MediumPurple'), row=1, col=1)

fig.add_trace(go.Scatter(x=predicted_prices.time,
                         y=y_test,
                         name='Truth',
                         marker_color='LightSkyBlue',
                         showlegend=False), row=2, col=1)

fig.add_trace(go.Scatter(x=predicted_prices.time,
                         y=y_pred,
                         name='Prediction',
                         marker_color='MediumPurple',
                         showlegend=False), row=2, col=1)

fig.show()
```
![newplot (6)](https://user-images.githubusercontent.com/78179650/110849041-5fccb300-8274-11eb-9498-3a36308f3880.png)  
(If only the chart is visible, try viewing this blog on our [themed webpage](https://aporter1350.github.io/)!)
As you can see, this particular model was able to accurately track changes in the price, but also consistently predicted much lower values of the weighted price than the actual ones. Additionally, the model appears to predict two prices at certain points in time. These anomalies can be attributed to the fact that this model only included an hour's worth of data. As a result, some of the features of XGBoosting might not perform as well as they otherwise would. Additionally, our weighting of the bid and ask prices to calculate the weighted price might not be a accurate representation of the actual trade prices of the IEX stock.

