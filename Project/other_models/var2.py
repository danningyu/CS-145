"""
Author: Vaishnavi Tipireddy
"""
import numpy as np
import pandas as pd
from util.constants import *
from util.functions import create_days_df
from calculate_test_error import calculate_test_error
from util.functions import transform_date_to_days_since
from statsmodels.tsa.api import VAR

from datetime import datetime

df = pd.read_csv("data/train.csv", parse_dates=True)
dft = pd.read_csv("data/test.csv", parse_dates=True)



def run_model(should_plot=False):
    df_train = pd.read_csv(TRAIN_DATA_FILENAME)
    df_train = transform_date_to_days_since(df_train)
    

    predictions = []

    for state in ALL_STATES:
        #only looking at New York
        #using date, confirmed, deaths, and people_hospitalized columns

        df_state = df_train.loc[df['Province_State'] == state]
    
        #df_state = df_state.loc[:, df_state.columns != 'ID']
        #df_state = df_state.loc[:, df_state.columns != 'Province_State']
        df_state = df_state[['Date', 'Confirmed', 'Deaths', 'Active']]
        df_state = df_state.dropna()
        
        #model
        model = VAR(df_state)
        lag_order = 4
        model_fitted = model.fit(lag_order)
        
        # Predict
        finput = df_state.values[-lag_order:]
        sept_pred = model_fitted.forecast(y=finput, steps = 26) 
        #results = pd.DataFrame(sept_pred, columns=df_state.columns)
        #print(results)

        predictions.append(sept_pred)

    return predictions


# print(predictions)

#reformat predictions and add forecast ID
predictions = run_model()
united = []
forecastIds = []
id = 0
for j in range(26):  # 9/1 to 9/26
    for i in range(len(ALL_STATES)):
    
        united.append((predictions[i][j]))
        forecastIds.append(id)
        id += 1

res = pd.DataFrame(united, columns=['Date','Confirmed', 'Deaths', 'Active'])

res = res[['Confirmed', 'Deaths']]
res = res.astype('int')
res.insert(0, "ForecastID", forecastIds, True) 


res.to_csv('SUBMISSION_FILE_NAME', index=False)
#print(res.head())
#print(res['Deaths'].dtype)

calculate_test_error()
