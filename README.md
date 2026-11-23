# Stock market prediction

Forecasts Amazon (AMZN) closing prices with a stacked LSTM. A learning exercise in time-series
preparation: windowing, scaling, and walk-forward evaluation on the most recent data.

## Approach (`Stock_Price.ipynb`)

1. Download about 5,000 days of AMZN prices with `yfinance`.
2. Use adjusted close, scale to [0, 1] with `MinMaxScaler`, and split 80% / 20% by time.
3. Build sliding windows: the previous 60 days predict the next day.
4. Model: two LSTM layers (128 and 64 units) followed by dense layers, trained with mean squared error.
5. Invert the scaling, compute RMSE on the held-out period and plot predictions against real prices.

This is not trading advice; next-day price prediction from price history alone has very limited
real-world value.

## Run

```bash
pip install tensorflow yfinance pandas numpy scikit-learn matplotlib jupyter
jupyter lab Stock_Price.ipynb
```

## Stack
Python · TensorFlow / Keras · pandas · scikit-learn · yfinance

## Licence
Apache 2.0
