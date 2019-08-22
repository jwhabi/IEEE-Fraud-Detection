# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 07:53:26 2019

@author: Jaideep.Whabi
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.metrics import roc_auc_score


df1=pd.read_csv("C:\\Users\\jaideep.whabi\\ieee-fraud-detection\\train_transaction.csv")

df2=pd.read_csv("C:\\Users\\jaideep.whabi\\ieee-fraud-detection\\train_identity.csv")

data=pd.merge(df1,df2,how='left',on='TransactionID')

del df1,df2

data.dtypes

data.head()

len(data.TransactionID.unique()) == len(data)

data.card2.unique()
data.card2.dtype
data.card4=data.card4.replace(np.nan,0)

data.card4=data.card4.astype('category')
data.card4=data.card4.cat.codes

for i in data.columns:
    data[i]=data[i].replace(np.nan,0)
    if(data[i].dtype=='object'):
        print(i,data[i].dtype)
        #data[i]=data[i].replace(np.nan,0)
        data[i]=data[i].astype('category')
        data[i]=data[i].cat.codes
        
data.isnull().values.any()        
data=data.drop('TransactionID',axis=1)

train_X, test_X, train_Y, test_Y = train_test_split( data.drop('isFraud',axis=1), data['isFraud'], test_size=1/7.0, random_state=0)

#X=data.drop('isFraud',axis=1)
#Y=data['isFraud']

scaler = StandardScaler()
# Fit on training set only.
scaler.fit(train_X)
# Apply transform to both the training set and the test set.
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)

pca = PCA(.95)
pca.fit(train_X)
pca.n_components_

#pca = PCA(n_components=10)
#principalComponents = pca.fit_transform(X)
#principalDf = pd.DataFrame(data = principalComponents)
train_X = pca.transform(train_X)
test_X = pca.transform(test_X)


#principalDf.head()

clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=9,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
#    tree_method='gpu_hist'
)

clf.fit(train_X, train_Y)

params = {'n_estimators':500,
    'max_depth':9,
    'learning_rate':0.05,
    'subsample':0.9,
    'colsample_bytree':0.9,
    "eval_metric": "auc"}
val_X=train_X[:10000]
partial_train_X=train_X[10000:]
val_Y=train_Y[:10000]
partial_train_Y=train_Y[10000:]

dtrain=xgb.DMatrix(partial_train_X,partial_train_Y)
dval=xgb.DMatrix(val_X,val_Y)
dtest=xgb.DMatrix(test_X)
clf=xgb.train(params,dtrain,num_boost_round=350,early_stopping_rounds=50, evals=[(dtrain,'train'),(dval,'test')],verbose_eval=True)

y_pred=clf.predict(dtest)
#y_pred=clf.predict_proba(dtest)

print('ROC AUC {}'.format(roc_auc_score(test_Y, y_pred)))
#ROC AUC 0.8420033677938177 - 10 rounds
#ROC AUC 0.8606789199156557 - 50 rounds
#ROC AUC 0.8889456436370466 - 200 rounds
#ROC AUC 0.8979520526895398 - 350 rounds