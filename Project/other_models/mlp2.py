"""
Author: Steven Luong
"""
from typing import Dict, Any

import numpy as np
import os.path
from sklearn.kernel_ridge import KernelRidge
import pandas as pd
from util.constants import *
from util.functions import transform_date_to_days_since, create_days_df
from calculate_test_error  import get_test_error
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import warnings

warnings.filterwarnings("ignore")

TRAINING_DATA = "../data/train.csv"
TEST_DATA = "../data/sept_data.csv"
NUM_DAYS_TO_PREDICT = 26  # for round 2 predictions
col_type = ['Confirmed','Deaths']
train_start = "04/12/2020"
train_end = "08/31/2020"
pred_start = "09/01/2020"
pred_end = "09/26/2020"

def tuning_models():
    models = []
    for solver_type in {'constant','adaptive','invscaling'}:
        for iter_num in {5000,10000}:
            for learning_rate in {0.001, 0.0001}:
                models.append(MLPRegressor(solver='lbfgs',max_iter=iter_num,learning_rate=solver_type,learning_rate_init=learning_rate))
    return models

def run_model(col_name,clf,should_plot=True,isDailyPred = False,max_day=5):
    df_train = pd.read_csv(TRAINING_DATA)
    df_train = transform_date_to_days_since(df_train)
    df_pred = create_days_df(start=train_start, end=train_end)
    df_pred_sept = create_days_df(start=pred_start, end=pred_end)
    df_actual = pd.read_csv("../data/sept_data.csv")

    testing_last_day_values = df_train[df_train.Date == 0]
    pred_last_day_values = df_train[df_train.Date == 141]
    predictions = []

    for state in ALL_STATES:
        df_state = df_train.loc[df_train['Province_State'] == state]
        x_training = df_state['Date'].to_numpy().reshape(-1,1)
        y_training = df_state[col_name].to_numpy().ravel()

        edt_y_training = np.concatenate(([0], y_training[:-1]))
        edt_y_training = y_training - edt_y_training
        edt_y_training[0] = 0
        edt_y_training[edt_y_training<0] = 0

        if isDailyPred:
            y_training = edt_y_training

        clf.fit(x_training[-max_day:],y_training[-max_day:])

        # # Plot fit curve versus actual data
        # fit_curve = clf.predict(df_pred['Date'].to_numpy().reshape(-1, 1))
        # if isDailyPred:
        #     fit_curve[0] += testing_last_day_values[testing_last_day_values['Province_State'] == state][col_name]
        #     for i in range(1, len(fit_curve)):
        #         fit_curve[i] += fit_curve[i - 1]

        sept_pred = clf.predict(df_pred_sept['Date'].to_numpy().reshape(-1, 1))
        # print(sept_pred)
        if isDailyPred:
            sept_pred[0] += pred_last_day_values[pred_last_day_values['Province_State'] == state][col_name]
            for i in range(1, len(sept_pred)):
                if sept_pred[i] > 0:
                    sept_pred[i] += sept_pred[i-1]
                else:
                    sept_pred[i] = sept_pred[i-1]

        predictions.append(sept_pred)

        # if should_plot and state == "California":
        if should_plot:
            plt.figure()
            plt.title(state)
            plt.xlabel("Days since 04-12-2020")
            plt.ylabel("Number of {0} cases".format(col_name))
            plt.plot(df_state['Date'], df_state[col_name], color="blue")
            plt.plot(df_pred_sept, sept_pred, color="orange")
            plt.plot(df_pred_sept, df_actual.loc[df_actual["Province_State"] == state][col_name],color="red")
            plt.show()

        # print("sept_pred {0}  = {1}".format(col_name, sept_pred))
        # df_actual = pd.read_csv("../data/sept_data.csv")
        # print("actual:", df_actual.loc[df_actual["Province_State"] == state])
    return predictions


models = {}
# models.update({'name': "KernelRidge poly degree 1", 'type': KernelRidge(kernel="poly", degree=1)})
# models.update({'name': "KernelRidge poly degree 2", 'type': KernelRidge(kernel="poly", degree=2)})
# models.append(KernelRidge(kernel="poly", degree=1))
# models.append(KernelRidge(kernel="poly", degree=2))
# models.append(KernelRidge(kernel="poly", degree=3))
# models.append(KernelRidge(kernel="sigmoid", degree=1))
# models.append(KernelRidge(kernel="sigmoid", degree=2))
# models.append(KernelRidge(kernel="sigmoid", degree=3))
# models.append(MLPRegressor(hidden_layer_sizes=(10,),solver='lbfgs',max_iter=10000,learning_rate='constant',learning_rate_init=0.001))
# models.append(MLPRegressor(solver='lbfgs',max_iter=10000,learning_rate='constant',learning_rate_init=0.01))
# models.append(MLPRegressor(solver='lbfgs',max_iter=10000,learning_rate='invscaling',learning_rate_init=0.01))
# models.append(MLPRegressor(solver='lbfgs',max_iter=10000,learning_rate='adaptive',learning_rate_init=0.01))
# models.append(MLPRegressor(solver='lbfgs',max_iter=10000,learning_rate='adaptive',learning_rate_init=0.01,max_fun=20000))
# models.append(MLPRegressor(solver='lbfgs',max_iter=10000,learning_rate='constant',learning_rate_init=0.01,max_fun=20000))
# models.append(MLPRegressor(solver='lbfgs',max_iter=10000,learning_rate_init=0.001))
# models.append(MLPRegressor(solver='lbfgs',max_iter=10000,learning_rate_init=0.0001))
# models.append(MLPRegressor(hidden_layer_sizes=(10,),solver='lbfgs',max_iter=100000,learning_rate='constant',learning_rate_init=0.001))
# models.append(MLPRegressor(hidden_layer_sizes=(10,),solver='adam',max_iter=100000,learning_rate='constant',learning_rate_init=0.001))
# models.append(MLPRegressor(
#     hidden_layer_sizes=(10,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
#     learning_rate='constant', learning_rate_init=0.09, power_t=0.5, max_iter=10000, shuffle=True,
#     random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
#     early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08))

best_mape = 10000
best_model = ""
best_max_day = 0

for num_day in {5,10,15,20}:
    for model in tuning_models():
        df = pd.DataFrame(columns=['ForecastID', 'Confirmed', 'Deaths'])
        print("Model {0} with max_day={1}".format(model, num_day))
        for m_type in col_type:
            predictions = run_model(m_type, model, should_plot=False, isDailyPred=True, max_day=num_day)
            united = []
            forecastIds = []
            id = 0
            for j in range(NUM_DAYS_TO_PREDICT):
                for i in range(len(ALL_STATES)):
                    united.append(int(predictions[i][j]))
                    forecastIds.append(id)
                    id += 1
            df['ForecastID'] = forecastIds
            df[m_type] = united
        df.to_csv("../data/submission.csv", index=False)
        error = get_test_error()
        print("MAPE=", error)
        if error < best_mape:
            best_mape = error
            best_model = model
            best_max_day = num_day

print("best model {0} maxday={1} with error={2} ".format(best_model,best_max_day,best_mape))
