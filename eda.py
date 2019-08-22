# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 11:25:11 2019

@author: Jaideep.Whabi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df1=pd.read_csv("C:\\Users\\jaideep.whabi\\ieee-fraud-detection\\train_transaction.csv")

df2=pd.read_csv("C:\\Users\\jaideep.whabi\\ieee-fraud-detection\\train_identity.csv")

data=pd.merge(df1,df2,how='left',on='TransactionID')

del df1,df2


data.dtypes

""" Categorical features:
    Categorical Features - Transaction
ProductCD
card1 - card6
addr1, addr2
P_emaildomain
R_emaildomain
M1 - M9
Categorical Features - Identity
DeviceType
DeviceInfo
id_12 - id_38 """

cat=['ProductCD','card1','card2','card3','card4','card5','card6','addr1','addr2','P_emaildomain','R_emaildomain',
     'M1','M2','M3','M4','M5','M6','M7','M8','M9','DeviceType','DeviceInfo',
     'id_12']

for i in cat:
    print(i)
    data[i]=data[i].astype('category')
    if(i=='id_12'):
        for j in list(range(13,39)):
            c= 'id_' + str(j)
            print(c)
            data[c]=data[c].astype('category')
missing_data=pd.DataFrame(columns=["Column","Percent Missing"])
for i in data.columns:
    print(i,": ",sum(data[i].isnull()) / len(data) * 100)
    missing_data.loc[len(missing_data)]=[i , sum(data[i].isnull()) / len(data) * 100]
    
sum(missing_data["Percent Missing"]<50)

for i in data.select_dtypes(include=['category']).columns:
   #if( data[i].dtype.name=='category'):
       data[i].hist()
       plt.save(i + ".png")
       plt.close()
       print(i)