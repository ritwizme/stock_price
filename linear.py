import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression


quandl.ApiConfig.api_key = 'dNKsx_NzCgQZAkusTZKG'
df=quandl.get_table('WIKI/PRICES',paginate=True)

df  = df[['adj_open','adj_high','adj_low','adj_close','adj_volume']]
df['hl_pct']=(df['adj_high']-df['adj_low'])/df['adj_close']*100.0
df['pct_change']=(df['adj_close']-df['adj_open'])/df['adj_open']*100.0
df = df[['adj_close','hl_pct','pct_change','adj_volume']]

forcast_col='adj_close'
df.fillna(-99999, inplace=True)

forcast_out=int(math.ceil(0.01*len(df)))
print(forcast_out)
df['label']=df[forcast_col].shift(-forcast_out)
df.dropna(inplace=True)


X = np.array(df.drop(['label'],1))
y = np.array(df['label'])
X = preprocessing.scale(X)
y = np.array(df['label'])

X_train, X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size = 0.2)

clf = LinearRegression()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)