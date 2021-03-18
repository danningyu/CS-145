# CS 145 Kaggle Project: COVID-19 Predictions

## Team 18 Submission
Group members: Khoa Le, Steven Luong, Vaishnavi Tipireddy, Danning Yu

The predictions for 12/7/2020 to 12/13/2020 are found in `Team18.csv`.

## Setup
To set up environment, run the following commands in a virtual environment containing Python 3.7.3 (higher versions of Python should also work):  
`pip install -r requirements.txt`

## Generate Predictions
All of the code to make the predictions is contained in `predict_exp_smoothing.py`. It has 1 required argument and 1 optional argument when invoked:
- `--predict {sept, dec}`: Required argument, predict for September or December
- `--plot`: Optional argument, make plots of model fits and forecasts

To get predictions for September (round 1):  
`python3 predict_exp_smoothing.py --predict sept`

To get predictions for December (round 2):  
`python3 predict_exp_smoothing.py --predict dec`

Caution: The output will overwrite the current contents of the output file, `Team18.csv`.  

You can add the `--plot` option to either command to have the script plot the predictions for cases or deaths for each individual state:  
`python3 predict_exp_smoothing.py --predict dec --plot`  
`python3 predict_exp_smoothing.py --predict sept --plot`  
## Data
- The file `data/train.csv` is the same as the `train.csv` provided to us on Kaggle
- The file `data/train_round2.csv` is similar to the data from `train_round2.csv` provided to us on Kaggle, but it only has ForecastID, Province_State, Date, Confirmed, and Deaths columns since those are the only ones our final model used. It also has data from 11/23 to 12/4 appended to it, taken directly from the [JHU dataset](https://github.com/CSSEGISandData/COVID-19).

## Other Models
Other time-series forecasting models that the team tried but had higher MAPE errors can be found in the `other_models` folder. The files were directly moved into the zip folder so there may be errors with undefined variables, missing modules, missing files, etc. These models can be run upon request. 

| Model Name             | Author              | MAPE for 9/1-9/26 | Relevant File      |
|------------------------|---------------------|-------------------|--------------------|
| Exponential Smoothing  | Danning Yu          | 2.04460           | `exp_smoothing.py` |
| Vector Auto Regression | Steven Luong        | 2.71              | `var.py`           |
| MLP Regressor          | Steven Luong        | 2.94              | `mlp2.py`          |
| Vector Auto Regression | Vaishnavi Tipireddy | 2.92              | `var2.py`          |
| Multilayer RNN (LSTM)  | Khoa Le             | 8.4               | `rnn.py`           |

We also tried other models like SVM, linear regression, and kernel regression. The results of those are described in our midterm report, and we abandoned them because the results were suboptimal and models specifically dedicated for time-series forecasting would work better.
