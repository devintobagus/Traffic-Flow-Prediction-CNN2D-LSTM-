import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import set_random_seed
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Conv2D, LSTM, Dense, Flatten, MaxPooling2D, TimeDistributed
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import time
from keras.regularizers import l2
from numpy import array
set_random_seed(1)
def window_data(df, window, target_col_number):
    """
    This function accepts the column number for the features (X) and the target (y).
    It chunks the data up with a rolling window of Xt - window to predict Xt.
    It returns two numpy arrays of X and y.
    """
    X = []
    y = []
    for i in range(len(df) - window - 1):
        features = df.iloc[i : (i + window),].T.values
        target = df.iloc[(i + window), target_col_number]
        #print(features)
        #print("----")
        #print(target)
        X.append(features)
        y.append(target)
    return np.array(X).astype(float), np.array(y).astype(np.float64).reshape(-1, 1)
def window_data_LSTM(df, window, target_col_number):
    """
    This function accepts the column number for the features (X) and the target (y).
    It chunks the data up with a rolling window of Xt - window to predict Xt.
    It returns two numpy arrays of X and y.
    """
    X = []
    y = []
    for i in range(len(df) - window - 1):
        features = df.iloc[i : (i + window),].values
        target = df.iloc[(i + window), target_col_number]
        #print(features)
        #print("----")
        #print(target)
        X.append(features)
        y.append(target)
    return np.array(X).astype(float), np.array(y).astype(np.float64).reshape(-1, 1)
def shaping(data):
    return data.reshape(data.shape[0], data.shape[1],data.shape[2], 1)
def NaN_initiate(df,percentage,target_columns=None):
    if target_columns==None:
        for col in df.columns:
            df.loc[df.sample(frac=percentage).index,col]=0
    else:
        df.loc[df.sample(frac=percentage).index,df.columns[target_columns]]=0
#Input data
df = pd.read_excel("Data_newalls.xlsx", index_col=[0], parse_dates=[0])
# NaN_initiate(df[900:996], 1, 8)
# NaN_initiate(df[900:920], 1, 2)
df=df.fillna(0)
# Split data
perc=0
length=844
df1=df[:length]
NaN_initiate(df1[:670],perc)
window_size=1
X, y=window_data(df1,window_size, 8)
X_LSTM, y_LSTM=window_data_LSTM(df1, window_size,8)

split = int(0.8 * len(X))
X_train = X[: split - 1]
X_test = X[split:]
X_LSTM=X_LSTM[:split-1]

y_train = y[: split - 1]
y_test = y[split:]
y_LSTM=y_LSTM[:split-1]

#Shaping
X_train=shaping(X_train)
X_test=shaping(X_test)


##Model
#Model 1
model1 = Sequential()
# model1.add(Dropout(0.2))
model1.add(Conv2D(7, (2,2), activation='relu',padding='same'))
model1.add(MaxPooling2D((2,2),padding='same'))
# model1.add(Dropout(0.2))
model1.add(TimeDistributed(Flatten()))
# model1.add(RepeatVector(1))
model1.add(LSTM(10, return_sequences=False))
# model1.add(Dropout(0.2))
model1.add(Flatten())
model1.add(Dense(1000, activation ='relu'))
model1.add(Dense(100))
# model1.add(Dropout(0.2))
model1.add(Dense(1))
model1.compile(loss='mse', optimizer='adam')

#Model 2
model2 = Sequential()
# model1.add(Dropout(0.2))
model2.add(Conv2D(7, (2,2), activation='relu',padding='same'))
model2.add(MaxPooling2D((2,2),padding='same'))
# model1.add(Dropout(0.2))
model2.add(Flatten())
model2.add(Dense(1000, activation ='relu'))
model2.add(Dense(100))
# model1.add(Dropout(0.2))
model2.add(Dense(1))
model2.compile(loss='mse', optimizer='adam')

#Model 1
model3 = Sequential()
model3.add(LSTM(100, return_sequences=False))
# model1.add(Dropout(0.2))
model3.add(Flatten())
model3.add(Dense(1000, activation ='relu'))
model3.add(Dense(100))
# model1.add(Dropout(0.2))
model3.add(Dense(1))
model3.compile(loss='mse', optimizer='adam')

##fit model
# model 1
history1 = model1.fit(X_train, y_train, epochs=50, batch_size= None, verbose=0)
history2 = model2.fit(X_train, y_train, epochs=50, batch_size= None, verbose=0)
history3 = model3.fit(X_LSTM, y_LSTM, epochs=50, batch_size= None, verbose=0)
#Prediction for first
forc1=[]
forc2=[]
forc3=[]
#Generate Data
app1=[]
app2=[]
model_range=673
# # #Model prediction
# #Reshaping input
k=0
# det_pred1=model1.predict(x_testT, verbose=0)
# det_pred1=det_pred1.reshape(det_pred1.shape[1],1,1)
j=0
for i in range(model_range,len(df.index)):
    x_input1=df.iloc[i:i+1,].T.values
    x_input2=df.iloc[i:i+1,].values
    x_input2=x_input2.reshape(1,x_input2.shape[0],x_input2.shape[1])
    x_input=shaping(x_input1.reshape(1,x_input1.shape[0],x_input1.shape[1]))
    yhat1=model1.predict(x_input,verbose=0)
    yhat2=model2.predict(x_input,verbose=0)
    yhat3=model3.predict(x_input2, verbose=0)
    forc1.append(yhat1)
    forc2.append(yhat2)
    forc3.append(yhat3)
    cnnlstm=array(forc1).reshape(-1,).astype(int)
    cnn=array(forc2).reshape(-1,).astype(int)
    lstm=array(forc3).reshape(-1,).astype(int)
    data=df.iloc[i:i+1,8].values
    app1.append(data)
    det=array(app1).reshape(-1,)
    # print(app)
    d1=pd.DataFrame({'Actual' : det, 'CNN-LSTM': cnnlstm, 'CNN': cnn, 'LSTM':lstm })
    d1.to_csv('dat.csv')
    time.sleep(1)

