
# coding: utf-8

# In[1]:


from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[23]:


start = '2006-01-01'
end = '2019-01-01'


# In[24]:


bac = data.DataReader("BAC", 'yahoo', start, end)
c = data.DataReader("C", 'yahoo', start, end)
gs = data.DataReader("GS", 'yahoo', start, end)
jpm = data.DataReader("JPM", 'yahoo', start, end)
ms = data.DataReader("MS", 'yahoo', start, end)
wfc = data.DataReader("WFC", 'yahoo', start, end)


# In[3]:


tickers = ['BAC', 'C', 'GS','JPM', 'MS','WFC' ]


# In[4]:


bank_stocks = pd.concat([bac,c,gs,jpm,ms,wfc], axis =1, keys=tickers)
bank_stocks.head()


# In[5]:


bank_stocks.columns.names = ['Bank Ticker','Stock Info']


# In[6]:


bank_stocks.xs(key = 'Close', axis = 1, level = 'Stock Info').max()


# In[8]:


returns = pd.DataFrame(
    
)
returns


# In[33]:


for tick in tickers:
    returns[tick + ' Returns'] = bank_stocks[tick]['Close'].pct_change()
returns.head()


# In[34]:


sns.pairplot(data = returns[1:])


# In[11]:


returns.idxmin()


# In[35]:


returns.idxmax()


# In[12]:


returns.std()


# In[36]:


returns.ix['2015-01-01':'2015-12-31'].std()


# In[13]:


sns.distplot(returns.ix['2015-01-01':'2015-12-31']['MSReturns'],color='green',bins=100)


# In[14]:


sns.distplot(returns.ix['2015-01-01':'2015-12-31']['CReturns'],color='red',bins=80);


# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

# Optional Plotly Method Imports
import plotly
import cufflinks as cf
cf.go_offline()


# In[16]:


bank_stocks.xs(key = 'Close', axis = 1, level = 'Stock Info').plot(figsize=(12,3))


# In[17]:


bank_stocks.xs(key = 'Close', axis = 1, level = 'Stock Info').loc['2008-01-01':'2008-12-31']['BAC'].rolling(30).mean().plot(figsize=(12,3))
bank_stocks.xs(key = 'Close', axis = 1, level = 'Stock Info').loc['2008-01-01':'2008-12-31']['BAC'].plot()


# In[18]:


sns.heatmap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)


# In[19]:


sns.clustermap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)


# In[37]:


close_corr = bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr()
close_corr.iplot(kind='heatmap',colorscale='rdylbu')


# In[21]:


bac[['Open', 'High', 'Low', 'Close']].ix['2015-01-01':'2016-01-01'].iplot(kind='candle')


# In[39]:


ms['Close'].ix['2015-01-01':'2016-01-01'].ta_plot(study='sma',periods=[13,21,55],title='Simple Moving Averages')


# In[40]:


bac['Close'].ix['2015-01-01':'2016-01-01'].ta_plot(study='boll')

