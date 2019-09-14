# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 15:39:20 2019

@author: Jaideep.Whabi
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.metrics import roc_auc_score,recall_score,precision_score,accuracy_score,confusion_matrix
#from imblearn.under_sampling import RandomUnderSampler
import random
from sklearn import metrics
import matplotlib.pyplot as plt
import re


train_id = pd.read_csv('C:\\Users\\jaideep.whabi\\ieee-fraud-detection\\train_identity.csv')
train_trn = pd.read_csv('C:\\Users\\jaideep.whabi\\ieee-fraud-detection\\train_transaction.csv')
test_id = pd.read_csv('C:\\Users\\jaideep.whabi\\ieee-fraud-detection\\test_identity.csv')
test_trn = pd.read_csv('C:\\Users\\jaideep.whabi\\ieee-fraud-detection\\test_transaction.csv')

id_cols = list(train_id.columns.values)
trn_cols = list(train_trn.drop('isFraud', axis=1).columns.values)

X_train = pd.merge(train_trn[trn_cols + ['isFraud']], train_id[id_cols], how='left')
#X_train = reduce_mem_usage(X_train)
X_test = pd.merge(test_trn[trn_cols], test_id[id_cols], how='left')
#X_test = reduce_mem_usage(X_test)

X_train_id = X_train.pop('TransactionID')
X_test_id = X_test.pop('TransactionID')
del train_id,train_trn,test_id,test_trn

all_data = X_train.append(X_test, sort=False).reset_index(drop=True)

#pca to reduce V columns
vcols = [f'V{i}' for i in range(1,340)]

sc = MinMaxScaler()

pca = PCA(n_components=2) #0.99
vcol_pca = pca.fit_transform(sc.fit_transform(all_data[vcols].fillna(-1)))

all_data['_vcol_pca0'] = vcol_pca[:,0]
all_data['_vcol_pca1'] = vcol_pca[:,1]
all_data['_vcol_nulls'] = all_data[vcols].isnull().sum(axis=1)

all_data.drop(vcols, axis=1, inplace=True)

#fillna for specific columns
all_data['card4'].fillna('unknown',inplace=True)
all_data['card6'].fillna('unknown',inplace=True)

all_data['P_emaildomain'].fillna('unknown',inplace=True)
all_data['R_emaildomain'].fillna('unknown',inplace=True)

#create features for date fields
import datetime

START_DATE = '2017-12-01'
startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
all_data['Date'] = all_data['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
all_data['_weekday'] = all_data['Date'].dt.dayofweek
all_data['_hour'] = all_data['Date'].dt.hour
#all_data['_day'] = all_data['Date'].dt.day

all_data['_weekday'] = all_data['_weekday'].astype(str)
all_data['_hour'] = all_data['_hour'].astype(str)
all_data['_weekday__hour'] = all_data['_weekday'] + all_data['_hour']

all_data.drop(['TransactionDT','Date'], axis=1, inplace=True)

#extract domain id (chrome etc)
all_data['_id_31_ua'] = all_data['id_31'].apply(lambda x: x.split()[0] if x == x else 'unknown')

#combine addr with other features
all_data['_P_emaildomain__addr1'] = all_data['P_emaildomain'] + '__' + all_data['addr1'].astype(str)
all_data['_card1__card2'] = all_data['card1'].astype(str) + '__' + all_data['card2'].astype(str)
all_data['_card1__addr1'] = all_data['card1'].astype(str) + '__' + all_data['addr1'].astype(str)
all_data['_card2__addr1'] = all_data['card2'].astype(str) + '__' + all_data['addr1'].astype(str)
all_data['_card12__addr1'] = all_data['_card1__card2'] + '__' + all_data['addr1'].astype(str)
all_data['_card_all__addr1'] = all_data['_card12__addr1'] + '__' + all_data['addr1'].astype(str)

#extract decimal and /10 remainder as seperate feature
all_data['_amount_decimal'] = ((all_data['TransactionAmt'] - all_data['TransactionAmt'].astype(int)) * 1000).astype(int)
all_data['_amount_decimal_len'] = all_data['TransactionAmt'].apply(lambda x: len(re.sub('0+$', '', str(x)).split('.')[1]))
all_data['_amount_fraction'] = all_data['TransactionAmt'].apply(lambda x: float('0.'+re.sub('^[0-9]|\.|0+$', '', str(x))))
all_data[['TransactionAmt','_amount_decimal','_amount_decimal_len','_amount_fraction']].head(10)


cols = ['ProductCD','card1','card2','card5','card6','P_emaildomain','_card_all__addr1']
#,'card3','card4','addr1','dist2','R_emaildomain'

# amount mean&std
for f in cols:
    all_data[f'_amount_mean_{f}'] = all_data['TransactionAmt'] / all_data.groupby([f])['TransactionAmt'].transform('mean')
    all_data[f'_amount_std_{f}'] = all_data['TransactionAmt'] / all_data.groupby([f])['TransactionAmt'].transform('std')
    all_data[f'_amount_pct_{f}'] = (all_data['TransactionAmt'] - all_data[f'_amount_mean_{f}']) / all_data[f'_amount_std_{f}']

# freq encoding
for f in cols:
    vc = all_data[f].value_counts(dropna=False)
    all_data[f'_count_{f}'] = all_data[f].map(vc)
    
print('features:', all_data.shape[1])

cat_cols = [f'id_{i}' for i in range(12,39)]
for i in cat_cols:
    if i in all_data.columns:
        all_data[i] = all_data[i].astype(str)
        all_data[i].fillna('unknown', inplace=True)

enc_cols = []
for i, t in all_data.loc[:, all_data.columns != 'isFraud'].dtypes.iteritems():
    if t == object:
        enc_cols.append(i)
        #df = pd.concat([df, pd.get_dummies(df[i].astype(str), prefix=i)], axis=1)
        #df.drop(i, axis=1, inplace=True)
        all_data[i] = pd.factorize(all_data[i])[0]
        #all_data[i] = all_data[i].astype('category')
print(enc_cols)

X_train = all_data[all_data['isFraud'].notnull()]
X_test = all_data[all_data['isFraud'].isnull()].drop('isFraud', axis=1)
Y_train = X_train.pop('isFraud')
del all_data

print([i for i in X_train.columns])
"""
['TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
 'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C3',
 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3',
 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M1', 'M2',
 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 
 'id_06', 'id_07', 'id_08', 'id_09', 'id_10', 'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 
 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25',
 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35',
 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', '_vcol_pca0', '_vcol_pca1',
 '_vcol_nulls', '_weekday', '_hour', '_weekday__hour', '_id_31_ua', '_P_emaildomain__addr1',
 '_card1__card2', '_card1__addr1', '_card2__addr1', '_card12__addr1', '_card_all__addr1',
 '_amount_decimal', '_amount_decimal_len', '_amount_fraction', '_amount_mean_ProductCD',
 '_amount_std_ProductCD', '_amount_pct_ProductCD', '_amount_mean_card1', '_amount_std_card1',
 '_amount_pct_card1', '_amount_mean_card2', '_amount_std_card2', '_amount_pct_card2', 
 '_amount_mean_card5', '_amount_std_card5', '_amount_pct_card5', '_amount_mean_card6',
 '_amount_std_card6', '_amount_pct_card6', '_amount_mean_P_emaildomain', 
 '_amount_std_P_emaildomain', '_amount_pct_P_emaildomain', '_amount_mean__card_all__addr1',
 '_amount_std__card_all__addr1', '_amount_pct__card_all__addr1', '_count_ProductCD',
 '_count_card1', '_count_card2', '_count_card5', '_count_card6', '_count_P_emaildomain',
 '_count__card_all__addr1']
"""

indexes=list(Y_train[Y_train==1].index)

i2=random.sample(list(Y_train[Y_train==0].index), len(indexes))

indexes = indexes + i2

X_train=X_train.loc[indexes]
Y_train=Y_train.loc[indexes]

Y_train.hist()

val_i=random.sample(list(Y_train.index), 2500)
pt_i=list(set(X_train.index) - set(val_i))
val_X=X_train.loc[val_i]
partial_X_train=X_train.loc[pt_i]
val_Y=Y_train.loc[val_i]
partial_Y_train=Y_train.loc[pt_i]
val_Y.hist()
partial_Y_train.value_counts()



import lightgbm as lgb

params={'learning_rate': 0.01,
        'objective': 'binary',
        'metric': 'auc',
        'num_threads': -1,
        'num_leaves': 256,
        'verbose': 1,
        'silent': False,
        'random_state': 42,
        'bagging_fraction': 1,
        'feature_fraction': 0.85
       }

oof_preds = np.zeros(X_train.shape[0])
sub_preds = np.zeros(X_test.shape[0])

clf = lgb.LGBMClassifier(**params, n_estimators=1500)
clf.fit(partial_X_train, partial_Y_train,eval_set=(val_X,val_Y))
oof_preds = clf.predict_proba(X_train, num_iteration=clf.best_iteration_)[:,1]
sub_preds = clf.predict_proba(X_test, num_iteration=clf.best_iteration_)[:,1]

fpr, tpr, thresholds = metrics.roc_curve(Y_train, oof_preds)
auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, label='ROC curve (area = %.3f)'%auc)
plt.legend()
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)

# Plot feature importance
feature_importance = clf.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
sorted_idx = sorted_idx[len(feature_importance) - 50:]
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(10,12))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

X_train.columns[np.argsort(-feature_importance)].values

submission = pd.DataFrame()
submission['TransactionID'] = X_test_id
submission['isFraud'] = sub_preds
submission.to_csv("C:\\Users\\jaideep.whabi\\ieee-fraud-detection\\submission14.csv", index=False)

#filename = #insert file path and name here
#pickle.dump(clf, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)