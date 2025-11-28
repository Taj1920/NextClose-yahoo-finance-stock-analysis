import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

#model training and evaluation

@st.cache_data
def train_models(X,y):
    #splitting into training and testing data
    Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.3,random_state=42)

    #model Training
    models = {"linear regression":LinearRegression(),"ridge":Ridge(alpha=1),"lasso":Lasso(alpha=0.01)}
    results = {}
    for model in models:
        models[model].fit(Xtrain,ytrain)
        ypred = models[model].predict(Xtest)
        r2 = r2_score(ytest,ypred)
        mse = mean_squared_error(ytest,ypred)
        mae =  mean_absolute_error(ytest,ypred)
        results[model] = {"model_obj":models[model],"r2":r2,"mse":mse,"mae":mae,"ypred":ypred}

    return results

def model_performance_report(results):
    res_df = pd.DataFrame({
        "Model":list(results.keys()),
        "R2 Score": [results[m]['r2'] for m in results],
        "MSE": [results[m]['mse'] for m in results],
        "MAE": [results[m]['mae'] for m in results]
    })
    
    return res_df
