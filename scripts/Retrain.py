import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

data = pd.read_csv('../data/data.csv')
X = []
y = []

for i in range(15, len(data)):
    close_values = data['Close'].iloc[i-15:i].values
    sma_value = data['SMA_7'].iloc[i]
    ema_value = data['EMA_7'].iloc[i]
    edit_count = data['edit_count'].iloc[i]
    sentiment = data['sentiment'].iloc[i]
    neg_sentiment = data['neg_sentiment'].iloc[i]
    feature_values = np.concatenate([close_values, [sma_value, ema_value,edit_count,sentiment,neg_sentiment]])
    X.append(feature_values)
    y.append(data['Close'].iloc[i])

X = np.array(X)
y = np.array(y)

scaler_x = StandardScaler()
X_normalized = scaler_x.fit_transform(X)
scaler_y = StandardScaler()
y_normalized = scaler_y.fit_transform(y.reshape(-1, 1))
X_normalized = X_normalized.reshape((X.shape[0], X.shape[1], 1))
total_size = len(X_normalized)
train_size = int(0.75 * total_size)
val_size = int(0.10 * total_size)
test_size = total_size - train_size - val_size
X_train, y_train = X_normalized[:train_size], y_normalized[:train_size]
X_val, y_val = X_normalized[train_size:train_size + val_size], y_normalized[train_size:train_size + val_size]
X_test, y_test = X_normalized[train_size + val_size:], y_normalized[train_size + val_size:]

model = Sequential()
model.add(LSTM(350, input_shape=(20, 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(250))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error',metrics=['mape'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, batch_size=30, validation_data=(X_val, y_val), callbacks=[early_stopping])
y_pred = model.predict(X_test)


scaler_x_filename = "../savedModels/scaler_x.pkl"
scaler_y_filename = "../savedModels/scaler_y.pkl"
joblib.dump(scaler_x, scaler_x_filename)
joblib.dump(scaler_y, scaler_y_filename)
model.save('../savedModels/model.h5')

y_pred_inverse = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
y_test_inverse = scaler_y.inverse_transform(y_test.reshape(-1, 1))
ErrorLstm = mean_absolute_percentage_error(y_test_inverse[:-1], y_pred_inverse[:-1])
print(ErrorLstm)
test_dates = data['Unnamed: 0'][-len(y_test):]
print(test_dates)

min_length = min(len(test_dates), len(y_pred_inverse), len(y_test_inverse))
test_dates = test_dates[-min_length:]
y_pred_inverse = y_pred_inverse[-min_length:]
y_test_inverse = y_test_inverse[-min_length:]


performance_df = pd.DataFrame({
    'Date': test_dates,
    'Model Prediction': y_pred_inverse.flatten(),
    'Actual Value': y_test_inverse.flatten()
})

performance_df.to_csv('../data/performance.csv')
