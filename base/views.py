from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
import yfinance as yf
from keras.models import load_model
import joblib
import numpy as np
import pandas as pd
import datetime
import os
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler


def home(request):
    return render(request, 'base/home.html')


def predictpage(request):
    try:
        filepath = os.path.join('data', 'data.csv')  
        data = pd.read_csv(filepath)
        model = load_model('savedModels/model.h5')
        scaler_x = joblib.load('savedModels/scaler_x.pkl')
        scaler_y = joblib.load('savedModels/scaler_y.pkl')
        
        close_prices = data['Close'].values[-15:]
        sma_7 = data['SMA_7'].values[-30:]
        ema_7 = data['EMA_7'].values[-30:]
        edit_count = data['edit_count'].values[-30:]
        sentiment = data['sentiment'].values[-30:]
        neg_sentiment = data['neg_sentiment'].values[-30:]
        
        results = []
        for i in range(15):
            sma_value_5 = sma_7[-1]
            ema_value_5 = ema_7[-1]
            edit_count5 = edit_count[-1]
            sentiment5 = sentiment[-1]
            neg_sentiment5 = neg_sentiment[-1]
            close_prices = close_prices.flatten()
            feature_values = np.concatenate([close_prices, [sma_value_5, ema_value_5, edit_count5, sentiment5, neg_sentiment5]])
            X = []
            X.append(feature_values)
            X = np.array(X).reshape(1, 20, 1)
            X = X.reshape(1, -1)
            X = scaler_x.transform(X)
            X = X.reshape((1, 20, 1))
            predictions = model.predict(X)
            predicted_value = scaler_y.inverse_transform(predictions)
            results.append(predicted_value[0][0])
            close_prices = np.append(close_prices, predicted_value)
            close_prices = close_prices[1:]
            sma_7 = np.append(sma_7, sma_7[-30:].mean())
            sma_7 = sma_7[1:]
            ema_7 = np.append(ema_7, ema_7[-30:].mean())
            ema_7 = ema_7[1:]
            edit_count = np.append(edit_count, edit_count[-30:].mean())
            edit_count = edit_count[1:]
            sentiment = np.append(sentiment, sentiment[-30:].mean())
            sentiment = sentiment[1:]
            neg_sentiment = np.append(neg_sentiment, neg_sentiment[-30:].mean())
            neg_sentiment = neg_sentiment[1:]

        start_date = datetime.date.today()
        dates = [(start_date + datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(15)]

        trends = []
        for i in range(1, len(results)):
            if results[i] > results[i - 1]:
                trends.append('up ⬆')
            else:
                trends.append('down ⬇')
        trends.insert(0, '-')

        predictions_with_dates = list(zip(dates, results, trends))

        advice = ""
        if trends.count('up ⬆') > trends.count('down ⬇'):
            advice = "The trend indicates a general upward movement. It might be a good time to invest."
        else:
            advice = "The trend indicates a general downward movement. It might be wise to wait or consider withdrawing."

        context = {
            'predictions_with_dates': predictions_with_dates,
            'investment_advice': advice,
        }

        return render(request, 'base/prediction.html', context)

    except Exception as e:
        return render(request, 'base/prediction.html', {'error': str(e)})

def modelHistory(request):
    filepath = os.path.join('data', 'performance.csv')  
    data = pd.read_csv(filepath)
    y_pred_inverse = data['Model Prediction']
    y_test_inverse = data['Actual Value']
    ErrorLstm = mean_absolute_percentage_error(y_test_inverse, y_pred_inverse)
    performance_data = list(zip(data['Date'], y_test_inverse, y_pred_inverse))

    context = {
        'data': performance_data,
        'Error': ErrorLstm,
        'accuracy': 100 - ErrorLstm
    }
    return render(request, 'base/modelhistory.html', context)

def bitcoinData(request):
    btc_ticker = yf.Ticker("BTC-USD")
    btc = btc_ticker.history(period="max")
    btc=btc[['Close']]
    context = {'data': btc[-30:]}
    return render(request, 'base/bitcoindata.html',context)
   
cryptos = [
    {'name': 'Bitcoin', 'ticker': 'BTC-USD', 'image_url': 'https://cryptologos.cc/logos/bitcoin-btc-logo.png'},
    {'name': 'Ethereum', 'ticker': 'ETH-USD', 'image_url': 'https://cryptologos.cc/logos/ethereum-eth-logo.png'},
    {'name': 'Binance Coin', 'ticker': 'BNB-USD', 'image_url': 'https://cryptologos.cc/logos/binance-coin-bnb-logo.png'},
    {'name': 'Solana', 'ticker': 'SOL-USD', 'image_url': 'https://cryptologos.cc/logos/solana-sol-logo.png'},
    {'name': 'Ripple', 'ticker': 'XRP-USD', 'image_url': 'https://cryptologos.cc/logos/xrp-xrp-logo.png'},
    {'name': 'Cardano', 'ticker': 'ADA-USD', 'image_url': 'https://cryptologos.cc/logos/cardano-ada-logo.png'}
]

def create_scalers(data):
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    scaler_x.fit(data)
    scaler_y.fit(data)
    
    return scaler_x, scaler_y

def fetch_and_prepare_data(ticker):
    coin_ticker = yf.Ticker(ticker)
    coin_data = coin_ticker.history(period="1mo")
    close_prices = coin_data[['Close']].iloc[-15:].values.reshape(-1, 1)  # Reshape to 2D array

    scaler_x, scaler_y = create_scalers(close_prices)
    model = load_model('savedModels/model_Generalize.h5')
    results = []
    
    for _ in range(10):
        X = scaler_x.transform(close_prices).reshape(1, 15, 1)  # Transform and reshape correctly
        predictions = model.predict(X)
        predicted_value = scaler_y.inverse_transform(predictions).reshape(-1, 1)  # Inverse transform and reshape
        results.append(predicted_value[0][0])
        close_prices = np.append(close_prices, predicted_value).reshape(-1, 1)
        close_prices = close_prices[1:]
    
    return results

def get_investment_advice(predictions):
    max_gain = 0
    best_crypto = ""
    up_trends = {}
    
    for crypto, data in predictions.items():
        gains = np.diff(data['prices'])
        up_days = np.sum(gains > 0)
        up_trends[crypto] = up_days
        
        if up_days > max_gain:
            max_gain = up_days
            best_crypto = crypto
    
    up_trend_cryptos = {k: v for k, v in up_trends.items() if v > 0}
    
    if not up_trend_cryptos:
        return "Currently, no coins are in an uptrend. It might not be the best time to invest."
    elif len(up_trend_cryptos) == 1:
        coin, days = list(up_trend_cryptos.items())[0]
        return f"The trend indicates that {coin} is in an uptrend for the consective {days} days. It might be better to invest until then."
    else:
        best_crypto, best_days = max(up_trend_cryptos.items(), key=lambda item: item[1])
        return f"The trend indicates that {best_crypto} is the best investment, with an uptrend for the consective {best_days} days. Other coins are also in an uptrend: {', '.join([f'{k} for {v} days' for k, v in up_trend_cryptos.items() if k != best_crypto])}."

def otherCrypto(request):
    predictions = {}
    
    for crypto in cryptos:
        name = crypto['name']
        ticker = crypto['ticker']
        predicted_prices = fetch_and_prepare_data(ticker)
        predictions[name] = {
            'dates': [(datetime.date.today() + datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(10)],
            'prices': predicted_prices
        }

    advice = get_investment_advice(predictions)

    context = {
        'cryptos': cryptos,
        'predictions': predictions,
        'investment_advice': advice
    }
    return render(request, 'base/otherCrypto.html', context)


def about(request):
    return render(request, 'base/about.html')