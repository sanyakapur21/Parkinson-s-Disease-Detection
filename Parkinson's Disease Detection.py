#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


df=pd.read_csv('/Users/sanyakapur/Downloads/parkinsons.data')
df.head()


# In[3]:


features=df.loc[:,df.columns!='status'].values[:,1:]
labels=df.loc[:,'status'].values


# In[4]:


print(labels[labels==1].shape[0], labels[labels==0].shape[0])


# In[5]:


scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels


# In[6]:


x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)


# In[7]:


model=XGBClassifier()
model.fit(x_train,y_train)


# In[8]:


y_pred=model.predict(x_test)
print(accuracy_score(y_test, y_pred)*100)


# In[ ]:




