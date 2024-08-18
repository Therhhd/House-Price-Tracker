import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.datasets import boston_housing

#Boston Dataset

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# Normalizing data 

mean = x_train.mean(axis=0)
std = x_train.std(axis=0)

x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

model = Sequential()

# LSTM layer and some Dropout regularisation
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(tf.keras.layers.Dropout(0.2))

# Adding eight more LSTM layers with dropout
for _ in range(13):
    model.add(LSTM(units=50, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))

# The output layer
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, epochs=10, batch_size=32)

test_loss = model.evaluate(x_test, y_test)

























