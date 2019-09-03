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



train_X, test_X, train_Y, test_Y = train_test_split( data.drop('isFraud',axis=1), data['isFraud'], test_size=1/7.0, random_state=0)
train_Y.value_counts()
#changing: undersampling to happen before feature selection
indexes=list(train_Y[train_Y==1].index)

i2=random.sample(list(train_Y[train_Y==0].index), len(indexes))

indexes = indexes + i2

train_X=train_X.loc[indexes]
train_Y=train_Y.loc[indexes]

train_Y.hist()



params = {'n_estimators':500,
    'max_depth':9,
    'learning_rate':0.05,
    'subsample':0.9,
    'colsample_bytree':0.9,
    "eval_metric": "auc"}
val_i=random.sample(list(train_Y.index), 5000)
val_X=train_X.loc[val_i]
partial_train_X=train_X.loc[val_i]
val_Y=train_Y.loc[val_i]
partial_train_Y=train_Y.loc[val_i]
val_Y.hist()
partial_train_Y.value_counts()

dtrain=xgb.DMatrix(partial_train_X,partial_train_Y)
dval=xgb.DMatrix(val_X,val_Y)
dtest=xgb.DMatrix(test_X)
clf=xgb.train(params,dtrain,num_boost_round=300,early_stopping_rounds=50, evals=[(dtrain,'train'),(dval,'test')],verbose_eval=True)

y_pred=clf.predict(dtest)
#y_pred=clf.predict_proba(dtest)

print('ROC AUC {}'.format(roc_auc_score(test_Y, y_pred)))
#ROC AUC 0.9010603919275391 - 300 rounds

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


#X=data.drop('isFraud',axis=1)
#Y=data['isFraud']
"""
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

"""
#principalDf.head()
""" DONOT USE
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
clf=xgb.train(params,dtrain,num_boost_round=10,early_stopping_rounds=50, evals=[(dtrain,'train'),(dval,'test')],verbose_eval=True)

y_pred=clf.predict(dtest)
#y_pred=clf.predict_proba(dtest)

print('ROC AUC {}'.format(roc_auc_score(test_Y, y_pred)))
#ROC AUC 0.8420033677938177 - 10 rounds
#ROC AUC 0.8606789199156557 - 50 rounds
#ROC AUC 0.8889456436370466 - 200 rounds
#ROC AUC 0.8979520526895398 - 350 rounds

#without pca
#ROC AUC 0.8597127557858613 - 10 rounds

#without pca and without imputation run TO-DO: code cleanup if this works
feature_importance=clf.get_score(importance_type='gain')
feature_importance=pd.DataFrame(data={'feature':list(feature_importance.keys()),'importance':list(feature_importance.values())})
feature_importance.sort_values(by='importance',ascending=False)
filters=feature_importance[feature_importance.importance>10]['feature']
filters.append(pd.Series('isFraud'))
data1=data[filters.append(pd.Series('isFraud'))]
train_X, test_X, train_Y, test_Y = train_test_split( data1.drop('isFraud',axis=1), data1['isFraud'], test_size=1/7.0, random_state=0)


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
clf=xgb.train(params,dtrain,num_boost_round=10,early_stopping_rounds=50, evals=[(dtrain,'train'),(dval,'test')],verbose_eval=True)

y_pred=clf.predict(dtest)
#y_pred=clf.predict_proba(dtest)

print('ROC AUC {}'.format(roc_auc_score(test_Y, y_pred)))
#only important features as per all first model
#ROC AUC 0.8522085343053979 - 10 rounds
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
"""
0    0.975558
Name: Accuracy, dtype: float64
0    [0.976529314358396, 0.8965853658536586]
Name: Precision Score, dtype: object
0    [0.998699194973493, 0.31965217391304346]
Name: Recall Score, dtype: object
"""
"""
#rus = RandomUnderSampler(random_state=42)
#X_res, y_res = rus.fit_resample(train_X, train_Y)


train_X, test_X, train_Y, test_Y = train_test_split( data1.drop('isFraud',axis=1), data1['isFraud'], test_size=1/7.0, random_state=0)


params = {'n_estimators':500,
    'max_depth':9,
    'learning_rate':0.05,
    'subsample':0.9,
    'colsample_bytree':0.9,
    "eval_metric": "auc"}
val_X=train_X[:1000]
partial_train_X=train_X[1000:]
val_Y=train_Y[:1000]
partial_train_Y=train_Y[1000:]

dtrain=xgb.DMatrix(partial_train_X,partial_train_Y)
dval=xgb.DMatrix(val_X,val_Y)
dtest=xgb.DMatrix(test_X)
clf=xgb.train(params,dtrain,num_boost_round=1000,early_stopping_rounds=50, evals=[(dtrain,'train'),(dval,'test')],verbose_eval=True)

y_pred=clf.predict(dtest)
#y_pred=clf.predict_proba(dtest)

print('ROC AUC {}'.format(roc_auc_score(test_Y, y_pred)))
#only important features + undersampling as per all first model
#ROC AUC 0.8669848039836963 - 10 rounds
#ROC AUC 0.91034886502859 - 300 rounds
#ROC AUC 0.9132674006601533 - 500 rounds
#ROC AUC 0.9136013001934172 - 1000 rounds
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
"""
1000 rounds (undersampling after feature selection)
0    0.844343
Name: Accuracy, dtype: float64
0    [0.8236931642437365, 0.8693373268438787]
Name: Precision Score, dtype: object
0    [0.8841301460823373, 0.8029045643153527]
Name: Recall Score, dtype: object
"""
