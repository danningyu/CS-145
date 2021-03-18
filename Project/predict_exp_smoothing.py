"""
Predict COVID-19 cases and deaths using Holt's model (a variant of exponential smoothing)
Author: Danning Yu
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import Holt

from constants import *
from functions import *

class COVID19Prediction:
    def __init__(self, args):
        """
        Initialize class with arguments passed in from command line
        """
        if args.plot:
            self.SHOULD_PLOT = True
        else:
            self.SHOULD_PLOT = False

        if args.predict == "sept":
            self.TRAINING_DATA = os.path.join("data", "train.csv")
            self.NUM_DAYS_TO_PREDICT = 26 # pred 9/1 to 9/26
            self.NUM_DAYS_TO_OUTPUT = 26
            self.ADJUST_FACTOR = 1
        else:
            self.TRAINING_DATA = os.path.join("data", "train_round2.csv")
            # We only use train data up to Friday (12/4) b/c there tends to be lower reporting on weekends
            # Thus, we predict 12/5-12/13 (9 days), but only will output 12/7-12/13 (7 days)
            self.NUM_DAYS_TO_PREDICT = 9 
            self.NUM_DAYS_TO_OUTPUT = 7
            self.ADJUST_FACTOR = 1.005 # There's a spike in COVID cases, so adjust all predictions uniformly upwards

    def create_holt_model(self, data, damped_trend=False, exponential=False, initialization_method="heuristic"):
        if damped_trend and exponential:
            print("ERROR: either damped or exponential, but not both!")
            exit(1)

        return Holt(data, exponential=exponential, damped_trend=damped_trend,initialization_method=initialization_method)

    def fit_holt_model(self, holt_model, smoothing_level, smoothing_trend, optimized=True):
        return holt_model.fit(smoothing_level=smoothing_level, smoothing_trend=smoothing_trend, optimized=optimized)

    def predict_holt_model(self, holt_model_fit, start_date, end_date):
        return holt_model_fit.predict(start_date, end_date)

    def run_model(self, col_name: str, state: str):
        """
        Extract data, run CV to get optimal hyperparameters, and then make predictions.

        args:
            col_name: the column to extract data from and predict for: either 'Confirmed' or 'Deaths'
            state: the state to make predictions for
        """
        # PART 1: EXTRACT DATA
        df_train = transform_date_to_days_since(pd.read_csv(self.TRAINING_DATA))
        df_state = df_train.loc[df_train['Province_State'] == state]
        X = pd.Series(df_state[col_name]).reset_index(drop=True)
        X = X[1:] # can't use exponential smoothing if any data point is 0 -> Wyoming had 0 cases on 4/12/2020


        # PART 2: TRAIN HPYERPARAMETERS
        VALIDATION_INTERVAL = 1
        START = df_state.tail(1)['Date'].values[0] - VALIDATION_INTERVAL # start of validation data
        
        damped_trend = False # can pass these in when creating model for different Holt variants
        exp_trend = False

        # Get optimal smoothing_level and smoothing_trend parameters
        train_data = X[ : START]
        test_data = X[START : START + VALIDATION_INTERVAL]
        best_st = -1
        best_sl = -1
        best_error = 100
                
        for st in [0.05, 0.1, 0.15, 0.2, 0.25, 0.275]:
            for sl in [0.425, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
                model = self.create_holt_model(train_data)
                fit_res = self.fit_holt_model(model, smoothing_level=sl, smoothing_trend=st)
                predict_data = self.predict_holt_model(fit_res, START, START + VALIDATION_INTERVAL - 1)

                err = abs_percent_error(ARR_OF_POINTS, predict_data.to_numpy().astype(int), test_data.to_numpy().astype(int))
                if abs(err) < abs(best_error):
                    best_error = err
                    best_st = st
                    best_sl = sl

        print(f"Best hyperparams for {state}, {col_name}: error={best_error:.4f}, st={best_st}, sl={best_sl}")
        

        # PART 3: MAKE ACTUAL PREDICTIONS
        model = self.create_holt_model(X) 
        fit_res = self.fit_holt_model(model, smoothing_level=best_sl, smoothing_trend=best_st)
        predict_data = self.predict_holt_model(fit_res, len(X), len(X) + self.NUM_DAYS_TO_PREDICT - 1)
        predict_train_data = self.predict_holt_model(fit_res, 0, len(X) - 1)
        predict_data = predict_data * self.ADJUST_FACTOR
    
        if self.SHOULD_PLOT:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6)) # width x height in inches

            axs[0].plot(predict_train_data.tail(25), color="orange", label="predictions")
            axs[0].plot(X.tail(25), color="blue", label="actual")
            axs[0].legend()
            axs[0].set(xlabel="Days since 04-12-2020", ylabel=col_name)
            axs[0].title.set_text("Last 25 days of training data for " + state)

            axs[1].plot(predict_data, color="orange", label="predictions", marker="o")
            axs[1].legend()
            axs[1].set(xlabel="Days since 04-12-2020", ylabel=col_name)
            axs[1].title.set_text("Predictions for " + state)
            data_index = predict_data.index
            for i, point in enumerate(predict_data):
                axs[1].annotate(f"{data_index[i]}", xy=(data_index[i], point))

            plt.show()

        return predict_data

    def get_predictions(self, col_name):
        """
        Get predictions for `col_name` for each of the 50 states.

        args:
            col_name: the column to make predictions for: either 'Confirmed' or 'Deaths'
        """
        predictions = []
        for state in ALL_STATES:
            pred = self.run_model(col_name, state)
            predictions.append(pred.to_numpy()[-1*self.NUM_DAYS_TO_OUTPUT:]) # extract the last NUM_DAYS_TO_OUTPUT of prediction data
        
        united = []
        for j in range(self.NUM_DAYS_TO_OUTPUT):
            for i in range(len(ALL_STATES)):   
                united.append(int(round(predictions[i][j]))) # round the floating point predictions
        
        return united
    
    def output_predictions(self):
        """
        Get predictions for confirmed and deaths for all 50 states and then write to CSV.
        """
        df = pd.DataFrame(columns=['ForecastID', 'Confirmed', 'Deaths'])

        forecastIDs = np.arange(0, len(ALL_STATES) * self.NUM_DAYS_TO_OUTPUT)
        df['ForecastID'] = forecastIDs

        df['Confirmed'] = self.get_predictions('Confirmed')
        df['Deaths'] = self.get_predictions('Deaths') 

        df.to_csv(SUBMISSION_FILE_NAME, index=False)

def main():
    parser = argparse.ArgumentParser(description="Predict COVID-19 cases and deaths for 50 states")
    parser.add_argument('--predict', choices={"sept", "dec"}, required=True, help="Predict for September or December")
    parser.add_argument('--plot', action="store_true", help="Make plots of model fits and forecasts")
    args = parser.parse_args()

    c = COVID19Prediction(args)
    c.output_predictions()

if __name__ == "__main__":
    main()
