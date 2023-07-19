from datetime import date, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Download gold prices from 2001 to present
df= pd.read_excel("/app/Prices.xlsx", sheet_name="Daily_Indexed", header=8, usecols="D:X")
df=df.drop(df.iloc[:,1:7],axis=1)
df=df.drop(df.iloc[:,2:], axis=1)
df=df.dropna()
df.rename(columns={'Name': 'Date', 'Indian rupee': 'Prices'}, inplace=True)

# LSTM model
# Load the saved LSTM model
model = load_model('lstm_model.h5')

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Display the title
st.title('GOLD Prediction per troy ounce in INR ')

# Display the descriptive statistics of the data
st.subheader('Data from 1979 - 2023')
st.write(df.describe())

# Visualize the open price
st.subheader('OPEN PRICE')
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(df.Date, df.Prices)
ax.set_xlabel('Date')
ax.set_ylabel('Price')
st.pyplot(fig)

# Scale the data
df1 = df[['Prices']]
scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(df1)

# Split the data
training_size = int(len(df1) * 0.70)
test_size = len(df1) - training_size
train_data, test_data = df1[0:training_size,:], df1[training_size:len(df1),:1]

# Create a function to create the dataset with a given time step
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Create the dataset
time_step = 4
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape the data for LSTM model
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Predict the train and test data
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform the predicted data
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Inverse transform the actual data
y_train = scaler.inverse_transform([y_train])
y_test = scaler.inverse_transform([y_test])

# Reshape the data for plotting
train_predict_plot = np.empty_like(df1)
train_predict_plot[:, :] = np.nan
train_predict_plot[time_step:len(train_predict)+time_step, :] = train_predict

test_predict_plot = np.empty_like(df1)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict)+(time_step)*2:len(df1)-2, :] = test_predict

# Plot the actual and predicted prices
st.subheader('Actual and Predicted Prices')
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(df.Date, scaler.inverse_transform(df1), label='Actual Prices')
ax.plot(df.Date[time_step:len(train_predict)+time_step], train_predict, label='Train Predictions')
ax.plot(df.Date[len(train_predict)+(time_step)*2:len(df1)-2], test_predict, label='Test Predictions')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

# Predicted Prices for Next 10 Days
st.subheader('Predicted Prices for Next 10 Days')

#PREDICTING THE NEXT 10 DAYS OPENING PRICE
x_input = test_data[681:].reshape(1, -1)
temp_input = list(x_input)
temp_input = temp_input[0].tolist()

from numpy import array

lst_output = []
n_steps = 150

i = 0
while i < 10:
    if len(temp_input) > n_steps:
        x_input = np.array(temp_input[-n_steps:])
        # print("{} day input {}".format(i, x_input))
        x_input = x_input.reshape((1, n_steps, 1))
        # print(x_input)
        yhat = model.predict(x_input, verbose=0)
        # print("{} day output {}".format(i, yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[-n_steps:]
        lst_output.extend(yhat.tolist())
        i = i+1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        # print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i = i+1

day_new = np.arange(1, 151)
day_pred = np.arange(151, 161)

# Plot the predicted prices
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(day_new, scaler.inverse_transform(df1[len(df1)-150:]))
ax.plot(day_pred, scaler.inverse_transform(lst_output))
ax.set_xlabel('Day')
ax.set_ylabel('Price')
st.pyplot(fig)



#   streamlit run "C:\Users\Hi\Desktop\DS_Projects\DS_WORK_PROJECTS\Gold_Price_Prediction\app.py"
