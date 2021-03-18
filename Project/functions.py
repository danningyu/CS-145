"""
Useful helper functions.
Author: Danning Yu
"""
import pandas as pd
import numpy as np
from datetime import datetime
from constants import *

SINGLE_POINT = "single_point"
ARR_OF_POINTS = "arr_of_points"


def abs_percent_error(calc_type, predicted, actual, calculate_avg=True) -> float:
    """
    Calculate absolute percent error for single or array of points

    args:  
        calc_type: either SINGLE_POINT or ARR_OF_POINTS  
        predicted: the predicted value, an int or np array of ints  
        actual: the actual value, an int or numpy array of ints  
        calculate_avg: for ARR_OF_POINTS mode only, whether to calculate the average  

    returns:  
        Absolute percentage error or mean absolute percentage error if calculate_avg is true
    """
    if predicted.dtype == np.float32 or predicted.dtype == np.float64:
        print("ERROR: predicted values must be ints and not floats")
        exit(1)
    if calc_type == SINGLE_POINT:
        # return 100 * abs((predicted - actual) / actual)
        return 100 * (predicted - actual) / actual
    elif calc_type == ARR_OF_POINTS:
        assert(len(predicted) == len(actual))
        total_err = np.sum(np.abs(np.divide(np.subtract(predicted, actual), actual)))
        # total_err = np.sum(np.divide(np.subtract(predicted, actual), actual))
        if calculate_avg:
            return 100 * total_err / len(predicted)
        else:
            return 100 * total_err

def transform_date_to_days_since(df):
    """
    Given Pandas dataframe with column Date as MM-DD-YYYY, convert to days since 04-12-2020
    """
    df['Date'] = (pd.to_datetime(df['Date']) - datetime(2020, 4, 12)).dt.days
    return df

def create_days_df(start, end):
    """
    Create Pandas dataframe with 'Date' column with values from start to end, but
    in days since 04-12-2020.
    """
    df = pd.DataFrame(columns=['Date'])
    df['Date'] = pd.date_range(start=start, end=end).to_series()
    df = transform_date_to_days_since(df).reset_index(drop=True)
    return df

def check_prediction_data(df_pred):
    """Check that the prediction CSV data is in the right format"""
    assert(df_pred['ForecastID'] is not None)
    assert(df_pred['Confirmed'] is not None)
    assert(df_pred['Deaths'] is not None)
    for col in list(df_pred):
        if df_pred[col].dtype == np.float32 or df_pred[col].dtype == np.float64:
            print(f"ERROR: predicted values for col {col} must be ints and not floats")
            exit(1)

def calculate_test_error(predicted_data, actual_data):
    """
    Caclulate MAPE given 2 files: predicted_data and actual_data.
    """
    df_actual = pd.read_csv(actual_data)
    df_pred = pd.read_csv(predicted_data)
    check_prediction_data(df_pred)

    error = abs_percent_error(ARR_OF_POINTS, 
                            df_pred['Confirmed'], 
                            df_actual['Confirmed'], 
                            calculate_avg=False)

    error += abs_percent_error(ARR_OF_POINTS, 
                            df_pred['Deaths'], 
                            df_actual['Deaths'], 
                            calculate_avg=False)

    error = error / (2*df_actual.shape[0])
    print(f"Mean absolute percentage error: {error}")
    return error
