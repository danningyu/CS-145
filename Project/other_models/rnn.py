
#!/usr/bin/env python
# coding: utf-8

"""
Author: Khoa Le
"""


# In[1]:


# ===>>> Module for preprocess data <<<===
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===>>> Module for learning <<<===
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


# In[2]:


# Read data
data = pd.read_csv("train.csv")

# List of states
states_name = data.Province_State.unique()

# Extract data 
states_data_confirmed = {}
states_data_deadth    = {}

for state_name in states_name:
    sample_data = data[data["Province_State"] == state_name]
    
    # Confirmed data
    sample      = sample_data[["Date","Confirmed"]].to_numpy()
    confirmed   = sample.T[1]
    
    states_data_confirmed[state_name] = confirmed
    
    # Deadth data
    
    sample      = sample_data[["Date","Deaths"]].to_numpy()
    deadth      = sample.T[1]
    
    states_data_deadth[state_name] = deadth


# In[3]:


number_day = len(states_data_confirmed[states_name[0]])


# In[4]:


# Below code is inspire after reading
# https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/

# split a univariate sequence into samples
def split_sequence(sequence, n_steps, n_output):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        mid_ix = i + n_steps
        end_ix = i + n_steps + n_output
        # check if we are beyond the sequence
        if end_ix > len(sequence):
            break
        # gather input and output parts of the patter
        seq_x, seq_y = sequence[i:mid_ix], sequence[mid_ix:end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.asarray(X), np.asarray(y)


# In[5]:


def predict_next_days( data, num_day = 26, num_epoch = 200 ):
    n_steps    = 14
    n_features = 1
    n_output   = 2
    
    X, y = split_sequence( data, n_steps, n_output )
    
    # define model
    model = Sequential()
    model.add(LSTM(32, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(32, activation='relu', return_sequences=True))
    model.add(LSTM(32, activation='relu'))
    model.add(Dense(20))
    model.add(Dense(n_output))
    model.compile(optimizer='adam', loss='mse')
    
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    
    # Trying to fix the error
    # ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type int).

    X = np.asarray(X).astype('float32')
    y = np.asarray(y).astype('float32')
    
    # fit model
    model.fit(X, y, epochs=num_epoch, verbose=0)
    
    # predict
    predict = data

    for i in range(num_day):
        x_input = predict[-n_steps:]
        x_input = x_input.reshape((1, n_steps, n_features))
        x_input = np.asarray(x_input).astype('float32')
       
        y_hat = model.predict( x_input, verbose= 0 )
        
        #print(predict.shape,' ',y_hat.shape)
        
        predict = np.append( predict, y_hat[0][0] )
        
    return predict


# In[6]:


output = np.asarray( [[i, 0, 0] for i in range(1300)] )

erros = []

count = 0

for state_name in states_name:
    print("Working with state ", count)
    
    data    = states_data_confirmed[state_name]
    predict = predict_next_days( data )
        
    plt.plot(predict, 'r--')
    plt.plot(data, 'b')
    plt.title(state_name + " confirmed")
    plt.show()
            
    predicted = predict[-26:]
    actual    = data[-26:]
    error = np.sum(np.abs(np.divide(np.subtract(predicted, actual), actual))) / len(predicted)
    print(error * 100)
    erros.append(error)
    
    for t in range(26):
        output[count + 50*t][1] = int(predicted[t])
    
    ### ====================== ###
    
    data    = states_data_deadth[state_name]
    predict = predict_next_days( data )
    
    plt.plot(predict, 'r--')
    plt.plot(data, 'b')
    plt.title(state_name + " death")
    plt.show()
    
    predicted = predict[-26:]
    actual    = data[-26:]
    error = np.sum(np.abs(np.divide(np.subtract(predicted, actual), actual))) / len(predicted)
    print(error * 100)
    erros.append(error)
    
    for t in range(26):
        output[count + 50*t][2] = int(predicted[t])
    
    ### ====================== ###
    
    count   = count + 1


# In[7]:


print(output)


# In[8]:


f = open("submission.csv", "w")
f.write("ForecastID,Confirmed,Deaths\n")
for i in range(1300):
    f.write(str(output[i][0]) + "," + str(output[i][1]) + "," + str(output[i][2]) + "\n")
f.close()


# In[9]:


#print(erros)
print("Average Error: ", sum(erros)/len(erros))
