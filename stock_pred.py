import yfinance as yf
import matplotlib.pyplot as plt 
import numpy as np

import numpy as np

class KalmanFilter:
    def __init__(self, A, B, H, Q, R, P0, x0):
        """
        A: state transition matrix
        B: control input matrix
        H: measurement matrix
        Q: process noise covariance
        R: measurement noise covariance
        P0: initial estimation error covariance
        x0: initial state estimate
        """
        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P0
        self.x = x0

    def predict(self, u=None):
        if u is None:
            u = np.zeros((self.B.shape[1],))

        # State prediction
        self.x = self.A @ self.x + self.B @ u
        # Covariance prediction
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x

    def update(self, z):
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state and covariance
        y = z - (self.H @ self.x)        # innovation
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P
        return self.x, K


def daily_returns(prices):

    returns = []
    for i in range(len(prices) - 1):
        returns.append((prices[i+1] - prices[i]) / prices[i])

    return returns

def rolling_volatility(returns, window_size):

    volatilities = []
    for i in range(len(returns) - window_size + 1):
        window = returns[i:i + window_size]
        volatility = np.std(window)
        volatilities.append(volatility)

    return volatilities

def kalman_filter():
    pass

dat = yf.Ticker("MSFT")

# download closing prices for MSFT over the last 3 years
data = yf.download("MSFT", period="3y")['Close']

# # plot closing prices
# plt.plot(data.index, data)
# plt.title("MSFT Closing Prices - Last 36 Months")
# plt.xlabel("Date")
# plt.ylabel("Closing Price (USD)")
# plt.grid()
# plt.show()

# # plot daily returns
returns = daily_returns(data.values)
# plt.plot(data.index[1:], returns)
# plt.title("MSFT Daily Returns - Last 36 Months")
# plt.xlabel("Date")
# plt.ylabel("Daily Return")
# plt.grid()
# plt.show()

# # plot rolling volatility with a w-day window
ws = 10
volatilities = rolling_volatility(returns, window_size=ws)
# plt.plot(data.index[ws:], volatilities)
# plt.title("MSFT Rolling Volatility (10-day window) - Last 36 Months")
# plt.xlabel("Date")
# plt.ylabel("Rolling Volatility")
# plt.grid()
# plt.show()

# Implement the Kalman Filter to estimate the stock price
n_A = 1
n_B = 1
n_H = 1
n_Q = 1
n_P = 1
A = np.eye(n_A) # State transition matrix
B = np.eye(n_B) # Control input matrix
H = np.eye(n_H) # Measurement matrix

Q = np.eye(n_Q) * 0.0001   # process noise
R = np.array([[.01]])   # measurement noise
P0 = np.eye(n_P) * 1.0
x0 = np.array(data.values[10])   # initial pos=0, vel=1

kf = KalmanFilter(A, B, H, Q, R, P0, x0)

# simulate
i = 0
preds = []
while i < len(volatilities):
    kf.predict([volatilities[i]])  # predict step
    y = data.values[i + ws - 1] + np.random.normal(0, R)  # measurement
    x_est, K = kf.update(y)
    preds.append(x_est[0])
    print("Estimate:", x_est, "Actual: ", data.values[i+ws], "Kalman Gain:", K)
    i=i+1

# plot results
plt.plot(data.index[ws:], data.values[ws:], label="Actual Price")
plt.plot(data.index[ws:], preds, label="Kalman Filter Estimate")
plt.title("Kalman Filter Stock Price Estimation - MSFT")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid()
plt.show()



