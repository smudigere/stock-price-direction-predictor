import numpy as np
import pandas as pd
import datetime
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data import TimeFrame
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SimpleRNN, GRU
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# API Keys Alpaca setup
API_KEY = ''
SECRET_KEY = ''
client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

def plot_stock_data(df):
  df['SMA5'] = df['close'].rolling(window=5).mean()
  df['SMA20'] = df['close'].rolling(window=20).mean()
  df['SMA50'] = df['close'].rolling(window=50).mean()

  fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                      vertical_spacing=0.03,
                      subplot_titles=('Candle Stick with SMAs', 'Volume'),
                      row_width=[0.7, 0.3])
  
  fig.add_trace(go.Candlestick(x=df.index,
                               open=df['open'],
                               high=df['high'],
                               low=df['low'],
                               close=df['close'],
                               name='CMLC'),
                row=1, col=1)
  
  fig.add_trace(go.Scatter(x=df.index, y=df['SMA5'],
                           line=dict(color='blue', width=1),
                           name='SMA 5'),
                row=1, col=1)
  
  fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'],
                           line=dict(color='orange', width=1),
                           name='SMA 20'),
                row=1, col=1)
  
  fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'],
                           line=dict(color='purple', width=1),
                           name='SMA 50'),
                row=1, col=1)
  
  # Add volume bars
  colors = ['green' if row['close'] > row['open'] else 'red'
            for index, row in df.iterrows()]
  fig.add_trace(go.Bar(x=df.index, y=df['volume'],
                       marker_color=colors,
                       name='Volume'),
                row=2, col=1)
  
  # Update layout
  fig.update_layout(title='SPY Stock Price and Volume Analysis',
                    yaxis_title='Price',
                    yaxis2_title='Volume',
                    xaxis_rangeslider_visible=False,
                    height=800)

  fig.show()


# Calculate technical indicators
def add_indicators(df):
  df['SMA5'] = df['close'].rolling(window=5).mean()
  df['SMA20'] = df['close'].rolling(window=20).mean()
  df['SMA50'] = df['close'].rolling(window=50).mean()

  # Calculate price changes
  df['Price_Change'] = df['close'].pct_change()

  # Calculate target (1 if tomorrow's price is higher, 0 if lower)
  df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)

  # Calculate RSI (Relative Strength Index)
  delta = df['close'].diff()
  gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
  rs = gain / loss
  df['RSI'] = 100 - (100 / (1 + rs))

  return df

# Prepare data for LSTM
def prepare_data(df, look_back=10):
  features = ['close', 'volume', 'SMA5', 'SMA20', 'SMA50', 'Price_Change', 'RSI']
  df = df.dropna()

  scaler = StandardScaler()
  scaled_data = scaler.fit_transform(df[features])

  X, y = [], []
  for i in range(look_back, len(scaled_data) - 1):
    X.append(scaled_data[i - look_back:i])
    y.append(df['Target'].iloc[i])

  return np.array(X), np.array(y)

# Create LSTM model
def create_lstm_model(input_shape):
  model = Sequential([
      LSTM(units=64, return_sequences=True, input_shape=input_shape),
      Dropout(0.2),
      LSTM(units=32, return_sequences=False),
      Dropout(0.2),
      Dense(units=16, activation='relu'),
      Dense(units=1, activation='sigmoid')
  ])

  model.compile(optimizer=Adam(learning_rate=0.02),
                loss='binary_crossentropy',
                metrics=['accuracy'])
  
  return model


# Create Simple RNN model
def create_rnn_model(input_shape):
    model = Sequential([
        SimpleRNN(units=64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        SimpleRNN(units=32, return_sequences=False),
        Dropout(0.2),
        Dense(units=16, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.02),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Create GRU model
def create_gru_model(input_shape):
  model = Sequential([
        GRU(units=64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(units=32, return_sequences=False),
        Dropout(0.2),
        Dense(units=16, activation='relu'),
        Dense(units=1, activation='sigmoid')
    ])

  model.compile(optimizer=Adam(learning_rate=0.02),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
  return model

if __name__ == '__main__':
    symbol = 'SPY'
    look_back = 10

    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        start=datetime.datetime(2021, 1, 1),
        end=datetime.datetime.now() - 1 * datetime.timedelta(days=1)
    )

    bars = client.get_stock_bars(request).df
    dataset = bars[bars.index.get_level_values(0) == symbol].copy()
    dataset.reset_index(inplace=True, drop=True)

    # Get data
    print('Fetching stock data...')
    df = dataset.copy()

    # Plot the data
    print('\nPlotting stock data...')
    plot_stock_data(df)

    # Add indicators
    df = add_indicators(df)

    # Prepare data for LSTM
    print('\nPreparing data for LSTM...')
    X, y = prepare_data(df, look_back)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create model
    print('\nCreating LSTM model...')
    lstm_model = create_lstm_model(input_shape=(look_back, X.shape[2]))

    # Create model
    print('\nCreating RNN model...')
    rnn_model = create_rnn_model(input_shape=(look_back, X.shape[2]))

    print('\nCreating GRU model...')
    gru_model = create_gru_model(input_shape=(look_back, X.shape[2]))


    history_lstm = lstm_model.fit(X_train, y_train,
                        batch_size=32,
                        epochs=100,
                        validation_split=0.1,
                        verbose=1
                        )

    history_rnn = rnn_model.fit(X_train, y_train,
                        batch_size=32,
                        epochs=100,
                        validation_split=0.1,
                        verbose=1
                        )

    history_gru = gru_model.fit(X_train, y_train,
                        batch_size=32,
                        epochs=100,
                        validation_split=0.1,
                        verbose=1
                        )

    # Evaluate model
    train_score = lstm_model.evaluate(X_train, y_train, verbose=0)
    test_score = lstm_model.evaluate(X_test, y_test, verbose=0)

    print('\nLSTM Model Performance Explanation:')
    print(f'Train Accuracy: {train_score[1]:.4f}')
    print(f'Test Accuracy: {test_score[1]:.4f}')

    # Make prediction for tomorrow
    last_sequence = X[-1:]
    tomorrow_pred = lstm_model.predict(last_sequence)[0][0]

    print('\nTomorrow\'s Prediction:')
    print(f'Probability of price increase: {tomorrow_pred:.2f}')
    print(f'Predicted direction: ', 'UP' if tomorrow_pred > 0.5 else 'DOWN')

    # Evaluate RNN model
    train_score = rnn_model.evaluate(X_train, y_train, verbose=0)
    test_score = rnn_model.evaluate(X_test, y_test, verbose=0)

    print('\nRNN Model Performance Explanation:')
    print(f'Train Accuracy: {train_score[1]:.4f}')
    print(f'Test Accuracy: {test_score[1]:.4f}')

    # Make prediction for tomorrow
    last_sequence = X[-1:]
    tomorrow_pred = rnn_model.predict(last_sequence)[0][0]

    print('\nTomorrow\'s Prediction:')
    print(f'Probability of price increase: {tomorrow_pred:.2f}')
    print(f'Predicted direction: ', 'UP' if tomorrow_pred > 0.5 else 'DOWN')

    # Evaluate GRU model
    train_score = gru_model.evaluate(X_train, y_train, verbose=0)
    test_score = gru_model.evaluate(X_test, y_test, verbose=0)

    print('\nGRU Model Performance Explanation:')
    print(f'Train Accuracy: {train_score[1]:.4f}')
    print(f'Test Accuracy: {test_score[1]:.4f}')

    # Make prediction for tomorrow
    last_sequence = X[-1:]
    tomorrow_pred = gru_model.predict(last_sequence)[0][0]

    print('\nTomorrow\'s Prediction:')
    print(f'Probability of price increase: {tomorrow_pred:.2f}')
    print(f'Predicted direction: ', 'UP' if tomorrow_pred > 0.5 else 'DOWN')
