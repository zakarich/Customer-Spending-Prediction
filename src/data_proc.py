import pickle 
import numpy as np 
import pandas as pd 
import os

def load_data(path):
    df = pd.read_csv(path)
    return df

def plot_info(df):
    print(df.info())
    print("\n ------------------------------------------- \n ")
    print(df.describe())
    print("\n ------------------------------------------- \n ")
    print("\n ------------------ Shape : ------------------ \n ")
    print(df.shape)
    print("\n ------------------------------------------- \n ")
    print("\n ------------------ Columns : ---------------- \n ")
    print(df.columns)
    print("\n ------------------ dtype : ------------------ \n ")
    print("\n ------------------------------------------- \n ")
    print(df.dtypes)

def date_split(df):
    df[["day", "month", "year"]] = df["invoice_date"].str.split("/", expand = True)
    return df
def save_splits(X_train,  X_test, y_train, y_test):
    file_Xtrain = 'dataset/Xtrain'
    file_Ytrain = 'dataset/Ytrain'
    file_Xtest = 'dataset/Xtest'
    file_Ytest = 'dataset/Ytest'
    pickle.dump(X_train,open(file_Xtrain,'wb'))
    pickle.dump(X_test,open(file_Xtest,'wb'))
    pickle.dump(y_train,open(file_Ytrain,'wb'))
    pickle.dump(y_test,open(file_Ytest,'wb'))
def missing_values_checker(df):
    count = 0
    for item in df.isnull().sum():
        if item != 0:
            count+=1
            
    if count != 0:
        print("the data has some missing values !!!!!")
    else:
        print("the data has no missing values ^_^ ")

    