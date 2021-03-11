import pyEX as p
import pandas as pd
import seaborn as sns
from jupyterthemes import jtplot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
jtplot.style(theme='onedork',context='talk',fscale=1.6)
from matplotlib.lines import Line2D
import talib as t #talib provides all kinds of tools for technical indicators
import re
import ast
import decimal
import math
from datetime import datetime
import re
from dateutil.rrule import rrule, DAILY
c = p.Client(api_token=YOUR API KEY)
sym='IEX'
col='close'
period=14 #RSI typically takes the first 14 days to generate an average change score
df = c.chartDF(symbol=sym, timeframe='1y')  #need a lot of data for a good plot
df.reset_index(inplace=True)
df.sort_values(by=['date'],inplace=True)
df['rsi']= t.RSI(df[col].values.astype(float), period)
df['ema']= t.EMA(df['close'].values.astype(float))
macd, macdsignal,macdhist=t.MACD(df['close'].values.astype(float))
df['macd']=macd
df['macdsignal']=macdsignal
df['macdhist']=macdhist
up,mid,low= t.BBANDS(df[col].values.astype(float),20)
df['uppr']=up
df['mid']=mid
df['low']=low
#Level 2 Data
L2=pd.read_csv('processedlevel2data.csv')
#drop na
L2.dropna(subset=['bidtimestamp', 'asktimestamp'],thresh=1,inplace=True)
L2.sort_values(by=['bidtimestamp','asktimestamp'],inplace=True)
L2['bid_rsi']= t.RSI(L2['bidprice'].values.astype(float), period)
L2['bidmacd'], L2['bidmacdSignal'], L2['bidmacdHist'] = t.MACD(L2['bidprice'].values.astype(float), fastperiod=12, slowperiod=26, signalperiod = 9)
L2['uppr'], L2['mid'], L2['low']=t.BBANDS(L2['bidprice'].values.astype(float),20)
bid=L2[['bidprice','bidtimestamp','bid_rsi','bidsize']]
bid.dropna(inplace=True)
bid.sort_values(by=['bidtimestamp'],inplace=True)
bid['bidtimestamp']=pd.to_datetime(bid['bidtimestamp'])

#Calculate rsi
fig=plt.figure(figsize=(15,10), constrained_layout=True)
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
#Bollinger band
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
plt.xlabel("Date", fontsize=15)
ax.legend(custom_lines, ['20 Day Mean Average','Upper Bound','Lower Bound'],loc='lower right',fontsize=15)
#ema
ax=sns.lineplot(x='date',y='ema',data=df,color='green')
sns.lineplot(x='date',y='close',ci=None,data=df, alpha=.5,color='orange')
plt.ylabel("Exponential Moving Average", fontsize=15)
plt.ylabel("Close", fontsize=15)
plt.xlabel("Date", fontsize=15)
ax.legend(custom_lines, ['EMA','Close'],loc='upper left',fontsize=15)
#macd
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


#Level 2 Data plots
#RSI for Level 2 data
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='blue', lw=4),
               Line2D([0], [0], color='red', lw=4)]
fig=plt.figure(figsize=(15,10), constrained_layout=True)
plt.rcParams['figure.constrained_layout.use'] = True
gs = gridspec.GridSpec(nrows=2, ncols=1, wspace=.2, hspace=.3)
ax0=fig.add_subplot(gs[0,0])
ax=sns.lineplot(x='bidtimestamp',y='bidprice',data=bid,ax=ax0,color='b')
#g=sns.lineplot(x='asktimestamp',y='askprice',data=ask,ax=ax0,color='r')
ax.set_title('IEX Price Level 2',fontsize=20)
plt.ylabel("Price", fontsize=15)
plt.xlabel("Date", fontsize=15)
#ax.legend(custom_lines, ['Bid','Ask'],loc='lower left',fontsize='large')
ax1=fig.add_subplot(gs[1,0],sharex=ax0)
ax=sns.lineplot(x='bidtimestamp',y='bid_rsi',data=bid,ax=ax1,color='b')
#sns.lineplot(x='asktimestamp',y='ask_rsi',ci=None,data=ask,ax=ax1,color='r')
ax.set_title('RSI Chart',fontsize=20)
plt.axhline(0, linestyle='--', alpha=0.1)
plt.axhline(20, linestyle='--', alpha=0.5)
plt.axhline(30, linestyle='--')
plt.ylabel("Relative Strength Index", fontsize=15)
plt.xlabel("Date", fontsize=15)
plt.axhline(70, linestyle='--')
plt.axhline(80, linestyle='--', alpha=0.5)
plt.axhline(100, linestyle='--', alpha=0.1)
#MACD for level 2
custom_lines = [Line2D([0], [0], color='blue', lw=4),
               Line2D([0], [0], color='red', lw=4)]
fig=plt.figure(figsize=(15,10), constrained_layout=True)
plt.rcParams['figure.constrained_layout.use'] = True
gs = gridspec.GridSpec(nrows=2, ncols=1, wspace=.2, hspace=.3)
ax0=fig.add_subplot(gs[0,0])
ax=sns.lineplot(x='bidtimestamp',y='bidprice',data=ask,ax=ax0,color='b')
ax.set_title('IEX Price Level 2',fontsize=20)
plt.ylabel("Bid Price", fontsize=15)
plt.xlabel("Date", fontsize=15)
ax1=fig.add_subplot(gs[1,0],sharex=ax0)
ax=sns.lineplot(x='bidtimestamp',y='bidmacd',ci=None,data=ask,ax=ax1,color='b')
ax.set_title('MACD Chart',fontsize=20)
plt.axhline(0, linestyle='--', alpha=0.1)
plt.ylabel("MACD Indicator", fontsize=15)
plt.xlabel("Date", fontsize=15)
#bollinger band for level 2
fig=plt.figure(figsize=(15,10), constrained_layout=True)
plt.rcParams['figure.constrained_layout.use'] = True
gs = gridspec.GridSpec(nrows=2, ncols=1, wspace=.2, hspace=.3)
ax0=fig.add_subplot(gs[0,0])
ax=sns.lineplot(x='date',y='bidprice',data=L2,ax=ax0)
ax.set_title('IEX Price Level 1',fontsize=20)
plt.ylabel("Bid Price", fontsize=15)
plt.xlabel("Date", fontsize=15)
ax1=fig.add_subplot(gs[1,0],sharex=ax0)
ax=sns.lineplot(x='date',y='mid',data=df,ax=ax1,color='orange')
sns.lineplot(x='date',y='uppr',data=df,ax=ax1,color='green')
sns.lineplot(x='date',y='low',data=df,ax=ax1,color='red')
ax.set_title('Bollinger Band',fontsize=20)
plt.xlabel("Date", fontsize=15)
ax.legend(custom_lines, ['20 Day Mean Average','Upper Bound','Lower Bound'],loc='lower right',fontsize=15)
