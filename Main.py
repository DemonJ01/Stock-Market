import math
import numpy as np
import pandas as pd
from datetime import date, timedelta
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import yfinance as yf

today = date.today()
date_today = today.strftime("%Y-%m-%d")
date_start = '2023-01-10'

# Getting Stocks quotes
stock = input("Enter stock name:")
stockname = stock
symbol = '^GSPC'

# Download stock data
df = yf.download(symbol, start=date_start, end=date_today)

# Download Nifty 50 data
nifty_symbol = '^NSEI'
nifty_df = yf.download(nifty_symbol, start=date_start, end=date_today)

# Taking a look at the shape of the dataset
print(df.shape)
df.head(5)

register_matplotlib_converters()
years = mdates.YearLocator()

# Plotting both stock and Nifty 50 on the same figure
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(16, 12))

# Plot for Stock
ax1.fill_between(df.index, 0, df['Close'], color='#b9e1fa')
ax1.legend([stockname], fontsize=12)
ax1.set_title(stockname + ' from ' + date_start + ' to ' + date_today, fontsize=16)
ax1.plot(df['Close'], color='#039dfc', label=stockname, linewidth=1.0)
ax1.set_ylabel('Stocks', fontsize=12)

# Plot for Nifty 50
ax2.fill_between(nifty_df.index, 0, nifty_df['Close'], color='#b9e1fa')
ax2.legend(['Nifty 50'], fontsize=12)
ax2.set_title('Nifty 50 from ' + date_start + ' to ' + date_today, fontsize=16)
ax2.plot(nifty_df['Close'], color='#E91D9E', label='Nifty 50', linewidth=1.0)
ax2.set_ylabel('Nifty 50', fontsize=12)

# Plot for Comparison
ax3.fill_between(df.index, 0, df['Close'], color='#b9e1fa')
ax3.fill_between(nifty_df.index, 0, nifty_df['Close'], color='#F0845C', alpha=0.5)  # Change the color as needed
ax3.legend([stockname, 'Nifty 50'], fontsize=12)
ax3.set_title(stockname + ' vs Nifty 50 from ' + date_start + ' to ' + date_today, fontsize=16)
ax3.plot(df['Close'], color='#039dfc', label=stockname, linewidth=1.0)
ax3.plot(nifty_df['Close'], color='#E91D9E', label='Nifty 50', linewidth=1.0)
ax3.set_ylabel('Comparison', fontsize=12)

# Format x-axis as years
ax1.xaxis.set_major_locator(years)
ax2.xaxis.set_major_locator(years)
ax3.xaxis.set_major_locator(years)

plt.show()

# Continue with the rest of your code...
train_df = df.filter(['Close'])
data_unscaled = train_df.values

# Get the number of rows to train the model on 80% of the data
train_data_length = math.ceil(len(data_unscaled) * 0.8)

# Transform features by scaling each feature to a range between 0 and 1
mmscaler = MinMaxScaler(feature_range=(0, 1))
np_data = mmscaler.fit_transform(data_unscaled)

sequence_length = 50

# Prediction Index
index_Close = train_df.columns.get_loc("Close")
print(index_Close)
# Split the training data into train and train data sets
# As a first step, we get the number of rows to train the model on 80% of the data
train_data_len = math.ceil(np_data.shape[0] * 0.8)

# Create the training and test data
train_data = np_data[0:train_data_len, :]
test_data = np_data[train_data_len - sequence_length:, :]


# The RNN needs data with the format of [samples, time steps, features]
# Here, we create N samples, sequence_length time steps per sample, and 6 features
def partition_dataset(sequence_length, train_df):
    x, y = [], []
    data_len = train_df.shape[0]
    for i in range(sequence_length, data_len):
        x.append(train_df[i - sequence_length:i, :])  # contains sequence_length values 0-sequence_length * columsn
        y.append(train_df[
                     i, index_Close])  # contains the prediction values for validation (3rd column = Close),  for single-step prediction

    # Convert the x and y to numpy arrays
    x = np.array(x)
    y = np.array(y)
    return x, y


# Generate training data and test data
x_train, y_train = partition_dataset(sequence_length, train_data)
x_test, y_test = partition_dataset(sequence_length, test_data)

# Print the shapes: the result is: (rows, training_sequence, features) (prediction value, )
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# Validate that the prediction value and the input match up
# The last close price of the second input sample should equal the first prediction value
print(x_test[1][sequence_length - 1][index_Close])
print(y_test[0])


model = Sequential()

neurons = sequence_length

# Model with sequence_length Neurons
# inputshape = sequence_length Timestamps
model.add(LSTM(neurons, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(neurons, return_sequences=False))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, batch_size=16, epochs=10)


y_pred_scaled = model.predict(x_test)
y_pred = mmscaler.inverse_transform(y_pred_scaled)
y_test_unscaled = mmscaler.inverse_transform(y_test.reshape(-1, 1))


# Mean Absolute Error (MAE)
MAE = mean_absolute_error(y_test_unscaled, y_pred)
print(f'Median Absolute Error (MAE): {np.round(MAE, 2)}')

# Mean Absolute Percentage Error (MAPE)
MAPE = np.mean((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled))) * 100
print(f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')

# Median Absolute Percentage Error (MDAPE)
MDAPE = np.median((np.abs(np.subtract(y_test_unscaled, y_pred)/ y_test_unscaled)) ) * 100
print(f'Median Absolute Percentage Error (MDAPE): {np.round(MDAPE, 2)} %')


# The date from which on the date is displayed
display_start_date = "2019-01-01"

# Add the difference between the valid and predicted prices
train = train_df[:train_data_length + 1]
valid = train_df[train_data_length:]
valid.insert(1, "Predictions", y_pred, True)
valid.insert(1, "Difference", valid["Predictions"] - valid["Close"], True)

# Zoom in to a closer timeframe
valid = valid[valid.index > display_start_date]
train = train[train.index > display_start_date]

# Visualize the data along with Nifty 50
# Visualize the data
fig, ax = plt.subplots(figsize=(16, 8), sharex=True)

plt.title("Predictions vs Ground Truth", fontsize=20)
plt.ylabel(stockname, fontsize=18)
plt.plot(train["Close"], color="#039dfc", linewidth=1.0)
plt.plot(valid["Predictions"], color="#E91D9E", linewidth=1.0)
plt.plot(valid["Close"], color="black", linewidth=1.0)
plt.legend(["Train", "Test Predictions", "Ground Truth"], loc="upper left")

# Fill between plotlines
# ax.fill_between(yt.index, 0, yt["Close"], color="#b9e1fa")
# ax.fill_between(yv.index, 0, yv["Predictions"], color="#F0845C")
# ax.fill_between(yv.index, yv["Close"], yv["Predictions"], color="grey")

# Create the bar plot with the differences
valid.loc[valid["Difference"] >= 0, 'diff_color'] = "#2BC97A"
valid.loc[valid["Difference"] < 0, 'diff_color'] = "#C92B2B"
plt.bar(valid.index, valid["Difference"], width=0.8, color=valid['diff_color'])

plt.show()


dif = valid['Close'] - valid['Predictions']
valid.insert(2, 'Difference', dif, True)
valid.tail(5)


# Get fresh data
df_new = df.filter(['Close'])

# Get the last N day closing price values and scale the data to be values between 0 and 1
last_days_scaled = mmscaler.transform(df_new[-sequence_length:].values)

# Create an empty list and Append past n days
X_test = []
X_test.append(last_days_scaled)

# Convert the X_test data set to a numpy array and reshape the data
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Get the predicted scaled price, undo the scaling and output the predictions
pred_price = model.predict(X_test)
pred_price_unscaled = mmscaler.inverse_transform(pred_price)

# Print last price and predicted price for the next day
price_today = round(df_new['Close'][-1], 2)
predicted_price = round(pred_price_unscaled.ravel()[0], 2)
percent = round(100 - (predicted_price * 100)/price_today, 2)

plus = '+'; minus = ''
print(f'The close price for {stockname} at {today} was {price_today}')
print(f'The predicted close price is {predicted_price} ({plus if percent > 0 else minus}{percent}%)')
