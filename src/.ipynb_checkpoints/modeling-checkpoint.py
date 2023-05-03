#split data
from sklearn.model_selection import train_test_split
#data modelling
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
import numpy as np



def split_data(df):
    y = df["price"] 
    X = df.drop('price',axis=1)
    return train_test_split(X, y, test_size=0.20, random_state = 0)

def scaling(X_train, X_test):
    scaler = StandardScaler() 
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def L_R(X_train, y_train):
    m1 = 'Linear Regression'
    lr = LinearRegression()
    return lr.fit(X_train, y_train)

def R_F(X_train, y_train):
    m2 = "Random Forest"
    rf = RandomForestRegressor(n_estimators=250, n_jobs=-1)
    return rf.fit(X_train,y_train)

def D_T(X_train, y_train):
    m3 = "Decision Tree"
    dtm = DecisionTreeRegressor(max_depth=5, min_samples_split=6, max_leaf_nodes=10)
    return dtm.fit(X_train,y_train)
    

def Ls_cv(X_train,y_train):
    ls_cv = LassoCV(alphas = None, cv = 10, max_iter = 100000)
    ls_cv.fit(X_train, y_train)
    alpha = ls_cv.alpha_
    ls = Lasso(alpha = ls_cv.alpha_)
    return ls.fit(X_train, y_train)

def Ridge_cv(X_train, y_train):
    alphas = np.random.uniform(0, 10, 50)
    r_cv = RidgeCV(alphas = alphas, cv = 10)
    r_cv.fit(X_train, y_train)
    alpha = r_cv.alpha_
    ri = Ridge(alpha = r_cv.alpha_)
    return ri.fit(X_train, y_train)