import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
import copy
import seaborn as sns
import tensorflow as tf
from sklearn.linear_model import LinearRegression

dataset_cols=['bike_count','hour','temp','humidity','wind','visibility','dew_pt_temp','radiation','rain','snow','functional']
df=pd.read_csv("SeoulBikeData.csv",encoding='ISO-8859-1').drop(['Date','Holiday','Seasons'],axis=1)

df.columns=dataset_cols
df['functional']=(df['functional']=='Yes').astype(int)
df=df[df['hour']==12]
df=df.drop(['hour'],axis=1)

for label in df.columns[1:]:
    plt.scatter(df[label],df['bike_count'])
    plt.title(label)
    plt.ylabel("Bike Count at Noon")
    plt.xlabel(label)
    plt.show()

#Based on the diagrams, delete the graphs that are not related
df=df.drop(['wind','visibility','functional'],axis=1)

#Train, Valid, Test Dataset
train, val, test = np.split(df.sample(frac=1),[int(0.6 *len(df)), int(0.8*len(df))])

def get_xy(dataframe,y_label,x_labels=None):
    dataframe=copy.deepcopy(dataframe)
    if x_labels is None:
        X=dataframe[[c for c in dataframe.colums if c!=y_label]].values
    else:
        if len(x_labels==1):
            X=dataframe[x_labels[0]].values.reshape(-1,1)
        else:
            X=dataframe[x_labels].values
    y=dataframe[y_label].values.reshape(-1,1)
    data=np.hstack((X,y))

    return data,X,y

_,X_train_temp,y_train_temp=get_xy(train,"bike count",x_labels=["temp"])
_,X_val_temp,y_val_temp=get_xy(val,"bike count",x_labels=["temp"])
_,X_test_temp,y_test_temp=get_xy(test,"bike count",x_labels=["temp"])

temp_reg=LinearRegression()
temp_reg.fit(X_train_temp,y_train_temp)

print(temp_reg.coef_,temp_reg.intercept_)
temp_reg.score(X_test_temp,y_test_temp)

plt.scatter(X_train_temp,y_train_temp,label='Data',color='blue')
x=tf.linspace(-20,40,100)
plt.plot(x,temp_reg.predict(np.array(x).reshape(-1,1)),label='Fit',color='red',linewidth=3)
plt.legend()
plt.title("Bikes vs Temp")
plt.ylabel("Number of bikes")
plt.xlabel("Temp")
plt.show()

#Multiple Linear Regression
df.columns








