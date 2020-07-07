# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:29:27 2020

@author: gandh
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 23 12:25:13 2020

@author: gandhi
"""
import itertools
import numpy as np
import pandas as pd
import os
from catboost.datasets import amazon
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score, log_loss
##Extracting the data
train_data, test_data =amazon()
train_data.head()
df= train_data
## Analysing the data
plt.figure()
plt.figure(figsize=(30,20))
for i in range(1,10):
    plt.subplot(5,2,i)
    plt.hist(df[df.columns[i]])
    plt.xlabel(df.columns[i])
    plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(30,20))
sns.heatmap(df.corr(),annot=True, cmap ='viridis', linewidth = 1)
plt.figure(figsize=(10,30))
sns.heatmap(df.corr(),annot=True,cmap='magma',linewidth=2)
## null data set
null_train = (train_data.isnull())
null_test =  test_data.isnull()
##Imbalanced Dataset graph
sns.countplot (train_data['ACTION'])


#duplicate columns
train_data.apply(lambda z: len(z.unique()))
##confirming the duplicated columns
from itertools import combinations
for feature_1,feature_2 in combinations(train_data.columns, 2):
    condition1=len(train_data.groupby([feature_1,feature_2]).size())==len(train_data.groupby([feature_1]).size())
    condition2=len(train_data.groupby([feature_1,feature_2]).size())==len(train_data.groupby([feature_2]).size())
    condition3=(train_data[feature_1].nunique()==train_data[feature_2].nunique())
    if (condition1 | condition2) & condition3:
        print(feature_1,feature_2)
        print('Duplicate Data')
##drop the duplicated columns
#new_train_data= train_data.drop('ROLE_CODE', axis =1)
#new_test_data = test_data.drop('ROLE_CODE' , axis =1)
## Encoding
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

target = 'ACTION'
train_columns = [x for x in train_data.columns if x not in [target, "ROLE_CODE"]]
Y = train_data["ACTION"].values
##
def encode_dataset(train, test, func, func_params = {}):
    dataset = pd.concat([train, test], ignore_index = True)
    dataset = func(dataset, **func_params)
    if isinstance(dataset, pd.DataFrame):
        new_train = dataset.iloc[:train.shape[0],:].reset_index(drop = True)
        new_test =  dataset.iloc[train.shape[0]:,:].reset_index(drop = True)
    else:
        new_train = dataset[:train.shape[0]]
        new_test =  dataset[train.shape[0]:]
    return new_train, new_test
def one_hot(dataset):
    ohe = OneHotEncoder(sparse=True, dtype=np.float32, handle_unknown='ignore')
    return ohe.fit_transform(dataset.values)
new_train, new_test = encode_dataset(train_data[train_columns], test_data[train_columns], one_hot)

print(new_train.shape, new_test.shape)
# Split into train & validation set
X_train, X_val, y_train, y_val = train_test_split(new_train, Y, train_size=0.8)
##LOGISTIC REGRESSION
 
model = LogisticRegression (penalty='l2',  
                C=1.0, dual = False,
                fit_intercept=True, 
                random_state=500,
                solver = 'liblinear',
                max_iter = 1000,)
model.fit(X_train, y_train)
       
##
def validation(model, X_test, y_test):
# Make predictions on test set
    y_pred=model.predict(X_test)
    y_pred=np.round(y_pred)
    
    # Confusion matrix
    print(confusion_matrix(y_test, y_pred))
    
    # AUC score
    y_pred_prob = model.predict_proba(X_test)
    print("AUC score: ", roc_auc_score(y_test, y_pred_prob[:,1]))
    

    # Accuracy, Precision, Recall, F1 score
    print(classification_report(y_test, y_pred))
    
    # Precision-Recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred)
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)
    plt.show()

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])
    plt.plot([0, 1], [0, 1],'k--')
    plt.plot(fpr, tpr, label='Neural Network')
    plt.xlabel('False Positive')
    plt.ylabel('True Positive ')
    plt.title('ROC Curve')
    plt.show()
    
##
validation(model, X_val, y_val)
stats = cross_validate(model, new_train, Y, groups=None, scoring='roc_auc', cv=5, n_jobs=2, return_train_score = True)
stats = pd.DataFrame(stats)
stats.describe().transpose()
## grouping
column_values=new_train_data["RESOURCE"].values
unique_values = np.unique(column_values)
print(unique_values)
df_group_MGR = new_train.groupby('MGR_ID')['RESOURCE'].nunique()
df_group_Title = new_train.groupby('ROLE_TITLE')['RESOURCE'].nunique()
##features and target
from sklearn.model_selection import train_test_split
x_features = new_train.values.drop("ACTION", axis = 1)
y_target = new_train["ACTION"]
x_features_train, x_features_valid, y_target_train, y_target_val = StratifiedKFold(x_features, y_target, train_size=0.8)
## determinig all the features for action 0
train_data_0 = new_train[new_train["ACTION"]==0].index
sample_0 = len(new_train[new_train["ACTION"]==0])
train_random_0= np.random.choice(train_data_0, sample_0, replace=False) 
train_0= train_data.loc[train_random_0]
new_train_0 = train_0.drop('ROLE_CODE' , axis = 1)
df_group_train_0 = new_train_0.groupby('ROLE_TITLE')['RESOURCE'].nunique()
##determining all the features for action 1

train_data_1 = new_train[new_train["ACTION"]==1].index
sample_1 = len(new_train[new_train["ACTION"]==1])
train_random_1= np.random.choice(train_data_1, sample_1, replace=False) 
train_1= train_data.loc[train_random_1]
new_train_1 = train_1.drop('ROLE_CODE' , axis = 1)
df_group_train_1 = new_train_1.groupby('ROLE_TITLE')['RESOURCE'].nunique()
## Concatinatig both the colummns
conc_train = pd.concat([new_train_1,new_train_0], ignore_index=True)
len(conc_train)


##pre processed dataset
##machine learning on basic
#to apply sampling techniques
##FLASK

