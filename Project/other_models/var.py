"""
Author: Steven Luong
"""
import numpy as np
import pandas as pd
from util.constants import *
import pandas
from util import *
from calculate_test_error import get_test_error
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.base.datetools import dates_from_str
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

TRAINING_DATA = "/data/train.csv"
TEST_DATA = "../data/sept_data.csv"
NUM_DAYS_TO_PREDICT = 26  # for round 2 predictions
col_type = ['Confirmed','Deaths']
train_start = "04/12/2020"
train_end = "08/31/2020"
pred_start = "09/01/2020"
pred_end = "09/26/2020"

df_train = pd.read_csv("data/train.csv")
for p in range(1,10):
    predictions = []
    model = None
    for state in ALL_STATES:
        # print(state)
        df_state = df_train.loc[df_train['Province_State'] == state]
        ds = dates_from_str(df_state['Date'])
        mdata = df_state[["Deaths", "Confirmed","Incident_Rate"]]
        last_day_data = mdata[-1:] #get the last day
        mdata.index = pandas.DatetimeIndex(ds)
        data = mdata.diff().dropna()
        model = VAR(data)
        results = model.fit(p)

        pred = results.forecast(data.values[-p:], 26)
        pred[0] = last_day_data
        # print(pred[0])
        for i in range(0, 26):
            if pred[i][0] < 0:
                pred[i][0] = 0
            if pred[i][1] < 0:
                pred[i][1] = 0
            pred[i] += pred[i-1]

        predictions.append(pred.T[0:2].T)

        # df_actual = pd.read_csv("data/sept_data.csv")
        # df_actual_confirm = df_actual.loc[df_actual['Province_State'] == state]["Confirmed"]
        # plt.plot(df_actual_confirm.to_numpy(), 'r')
        # plt.plot(pred.T[1:2].T, 'b--')
        # plt.show()
    df = pd.DataFrame(columns=['ForecastID', 'Confirmed', 'Deaths'])
    for col in {"Deaths", "Confirmed"}:
        united = []
        forecastIds = []
        id = 0
        for j in range(NUM_DAYS_TO_PREDICT):
            for i in range(len(ALL_STATES)):
                temp = predictions[i][j]
                value = temp[1]
                if (col == "Deaths"):
                    value = temp[0]
                united.append(int(value))
                forecastIds.append(id)
                id += 1
        # df['ForecastID'] = forecastIds
        df[col] = united
    df['ForecastID'] = forecastIds
    df.to_csv("./data/submission.csv", index=False)
    error = get_test_error()
    print("P={0} - error={1}".format(p,error))
