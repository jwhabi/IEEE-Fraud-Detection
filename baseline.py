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
from sklearn.metrics import roc_auc_score,recall_score,precision_score,accuracy_score,confusion_matrix
#from imblearn.under_sampling import RandomUnderSampler
import random

df1=pd.read_csv("C:\\Users\\jaideep.whabi\\ieee-fraud-detection\\train_transaction.csv")

df2=pd.read_csv("C:\\Users\\jaideep.whabi\\ieee-fraud-detection\\train_identity.csv")

data=pd.merge(df1,df2,how='left',on='TransactionID')

del df1,df2


for i in data.columns:
    #data[i]=data[i].replace(np.nan,0) # uncomment if using pca
    if(data[i].dtype=='object'):
        print(i,data[i].dtype)
        #data[i]=data[i].replace(np.nan,0)
        data[i]=data[i].astype('category')
        data[i]=data[i].cat.codes
        
#data.isnull().values.any()        
data=data.drop('TransactionID',axis=1)

#use if undersampling prior to split
#indexes=list(data[data.isFraud==1].index)
#
#i2=random.sample(list(data[data.isFraud==0].index), len(indexes))
#
#indexes = indexes + i2
#
#data=data.loc[indexes]
#
#data.isFraud.value_counts()

#pca for V columns
vcols = [f'V{i}' for i in range(1,340)]

sc = StandardScaler()

pca = PCA(n_components=2) #0.99
vcol_pca = pca.fit_transform(sc.fit_transform(data[vcols].fillna(-1)))

data['_vcol_pca0'] = vcol_pca[:,0]
data['_vcol_pca1'] = vcol_pca[:,1]
data['_vcol_nulls'] = data[vcols].isnull().sum(axis=1)

data.drop(vcols, axis=1, inplace=True)

train_X, test_X, train_Y, test_Y = train_test_split( data.drop('isFraud',axis=1), data['isFraud'], test_size=0.05, random_state=0)
train_Y.value_counts()
test_Y.value_counts()

#changing: undersampling to happen before feature selection
indexes=list(train_Y[train_Y==1].index)

i2=random.sample(list(train_Y[train_Y==0].index), len(indexes))

indexes = indexes + i2

train_X=train_X.loc[indexes]
train_Y=train_Y.loc[indexes]

train_Y.hist()

#feature scaling decreases score
#mean = train_X.mean(axis=0)
#train_X -= mean
#std = train_X.std(axis=0)
#train_X /= std
#
#test_X -= mean
#test_X /= std

params = {'n_estimators':500,
    'max_depth':9,
    'learning_rate':0.05,
    'subsample':0.9,
    'colsample_bytree':0.9,
    "eval_metric": "auc"}
val_i=random.sample(list(train_Y.index), 1000)
pt_i=list(set(train_X.index) - set(val_i))
val_X=train_X.loc[val_i]
partial_train_X=train_X.loc[pt_i]
val_Y=train_Y.loc[val_i]
partial_train_Y=train_Y.loc[pt_i]
val_Y.hist()
partial_train_Y.value_counts()

#val_X=train_X[:1000]
#partial_train_X=train_X[1000:]
#val_Y=train_Y[:1000]
#partial_train_Y=train_Y[1000:]


dtrain=xgb.DMatrix(partial_train_X,partial_train_Y)
dval=xgb.DMatrix(val_X,val_Y)
dtest=xgb.DMatrix(test_X)
clf=xgb.train(params,dtrain,num_boost_round=450,early_stopping_rounds=50, evals=[(dtrain,'train'),(dval,'test')],verbose_eval=True)
clf.attributes()

y_pred=clf.predict(dtest)
#y_pred=clf.predict_proba(dtest)

print('ROC AUC {}'.format(roc_auc_score(test_Y, y_pred)))
#ROC AUC 0.9010603919275391 - 300 rounds
#ROC AUC 0.9474167644773602 - if undersample before split
#ROC AUC 0.9491605651980637 - undersample after split (different split ratio)
#ROC AUC 0.9515089718093055 - if undersample before split (ultra small test set)
#ROC AUC 0.9508173314349682 - undersample after split and feature scaling

#ROC AUC 0.9600422200727456 - undersample after split 600 iterations

#ROC AUC 0.9497311138617524 - pca on v columns and then same stuff, 300 iter
#ROC AUC 0.9541357072760409 - pca on v then 450 iter
#ROC AUC 0.9546193042881589 - pca on v then 500 iter
"""
feature_importance=clf.get_score(importance_type='gain')
feature_importance=pd.DataFrame(data={'feature':list(feature_importance.keys()),'importance':list(feature_importance.values())})
feature_importance.sort_values(by='importance',ascending=False)
filters=feature_importance[feature_importance.importance>0.8]['feature']
filters.append(pd.Series('isFraud'))
data1=data[filters.append(pd.Series('isFraud'))]
train_X, test_X, train_Y, test_Y = train_test_split( data1.drop('isFraud',axis=1), data1['isFraud'], test_size=1/7.0, random_state=0)


params = {'n_estimators':500,
    'max_depth':9,
    'learning_rate':0.05,
    'subsample':0.9,
    'colsample_bytree':0.9,
    "eval_metric": "auc"}
val_X=train_X[:5000]
partial_train_X=train_X[5000:]
val_Y=train_Y[:5000]
partial_train_Y=train_Y[5000:]

dtrain=xgb.DMatrix(partial_train_X,partial_train_Y)
dval=xgb.DMatrix(val_X,val_Y)
dtest=xgb.DMatrix(test_X)
clf2=xgb.train(params,dtrain,num_boost_round=300,early_stopping_rounds=50, evals=[(dtrain,'train'),(dval,'test')],verbose_eval=True)

y_pred=clf2.predict(dtest)
#y_pred=clf.predict_proba(dtest)

print('ROC AUC {}'.format(roc_auc_score(test_Y, y_pred)))
#only important features as per all first model
#ROC AUC 0.9201510578249473 - 300 rounds
pred=y_pred
pred=np.where(pred>0.5,1,0)
accuracy = pd.DataFrame(columns=['Accuracy'])

acu=accuracy_score(test_Y, pred)
rec=recall_score(test_Y, pred,average=None)
pre=precision_score(test_Y, pred,average=None)
accuracy=accuracy.append([{'Accuracy':acu,'Recall Score': rec, 'Precision Score': pre}])
for c in accuracy.columns:
    print(accuracy[c])
"""
#trying lightgbm
    
import lightgbm as lgb

params={'learning_rate': 0.01,
        'objective': 'binary',
        'metric': 'auc',
        'num_threads': -1,
        'num_leaves': 256,
        'verbose': 1,
        'random_state': 42,
        'bagging_fraction': 1,
        'feature_fraction': 0.85
       }

lgb_train = lgb.Dataset(partial_train_X,partial_train_Y)
lgb_eval = lgb.Dataset(val_X,val_Y, reference=lgb_train)
lgb_test = lgb.Dataset(test_X)
clf = lgb.train(params,
                lgb_train,
                num_boost_round=450,
                valid_sets=lgb_eval)#,
                #early_stopping_rounds=20)

y_pred=clf.predict(test_X)
print('ROC AUC {}'.format(roc_auc_score(test_Y, y_pred)))
#ROC AUC 0.9397014160949145 - 300 rounds lgbm
#ROC AUC 0.9377951724133072 - pca on v columns then lgbm
#ROC AUC 0.9574895013409965 - pca on v not undersample

"""Test data read and predict"""

df1=pd.read_csv("C:\\Users\\jaideep.whabi\\ieee-fraud-detection\\test_transaction.csv")

df2=pd.read_csv("C:\\Users\\jaideep.whabi\\ieee-fraud-detection\\test_identity.csv")

data=pd.merge(df1,df2,how='left',on='TransactionID')

del df1,df2

for i in data.columns:
    #data[i]=data[i].replace(np.nan,0) # uncomment if using pca
    if(data[i].dtype=='object'):
        print(i,data[i].dtype)
        #data[i]=data[i].replace(np.nan,0)
        data[i]=data[i].astype('category')
        data[i]=data[i].cat.codes

s_data=data.drop('TransactionID',axis=1)
#s_data -= mean
#s_data /= std
vcol_pca = pca.transform(sc.fit_transform(s_data[vcols].fillna(-1)))

s_data['_vcol_pca0'] = vcol_pca[:,0]
s_data['_vcol_pca1'] = vcol_pca[:,1]
s_data['_vcol_nulls'] = data[vcols].isnull().sum(axis=1)

s_data.drop(vcols, axis=1, inplace=True)


#dtest=xgb.DMatrix(s_data)

y_pred=clf.predict(s_data)

submit= pd.DataFrame(data={'TransactionID':data['TransactionID'],'isFraud':y_pred})
submit.to_csv("C:\\Users\\jaideep.whabi\\ieee-fraud-detection\\submission10.csv",index=False)

#submission 1 is undersampling prior to split with split ratio as 1/7.0... 
#submission 2 is undersampling post split with split ratio as 0.05
#submission 3 is undersampling prior to split with split ratio as 0.05
#for submission 2 and 3 training data is same ~38k but test data for submission 3 is 
#only 2000 rows.... and only a slightly better test result...check kaggle submission
#submission 4 is undersampling after split with feature scaling
#submission 5 is undersampling after split while training with 600 iterations.
#submission6 same as submission2 but 300 iterations of xgboost
#submission7 is lgbm 300 iterations
#submission 8 is pca on vs, undersample then 450 xgboost
#submission 9 is pca on vs , undersample the 500 pca
#submission 10 is pca on vs, no undersampling, lgbm 450 rounds
#submission 11 is first trial new approach after eda, adding new features
# submission 12 is new approach with undersampling and 500 rounds of lgbm
#submission 13 is new approach with undersampling and val data added for 1500 lgbm rounds
# submission 14 same as 13 but 3000 rounds with early stopping at 2655
###################################################################

#