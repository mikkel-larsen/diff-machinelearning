import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from models import Bachelier
from models import BlackScholes
from models import Basket

class PolyReg:
    def __init__(self, x, y, p):
        self.x = x
        self.y = y
        self.p = p
        self.poly_features = PolynomialFeatures(degree=p)
        self.X = self.poly_features.fit_transform(x)
        self.params = None

    def fit(self, lmbda=0):
        n = np.shape(self.X)[1]
        X_t = np.transpose(self.X)
        self.params = np.dot(np.linalg.solve(np.dot(X_t, self.X) + np.identity(n) * lmbda, X_t), self.y)
        return self.params

    def fit_differential(self, Z, lmbda=None):
        if lmbda is None:
            lmbda = np.dot(np.transpose(ypoints), ypoints) / np.dot(np.transpose(Z[0, :, :]), Z[0, :, :])

        X_t = np.transpose(self.X)
        powers = self.poly_features.powers_
        D = np.transpose(np.transpose(powers)[:, :, np.newaxis] * np.transpose(self.X), axes=(0, 2, 1)) / \
            np.transpose(self.x[:, :, np.newaxis], axes=(1, 0, 2))
        D_t = np.transpose(D, axes=(0, 2, 1))
        p1 = np.dot(X_t, self.X) + lmbda * np.sum(np.matmul(D_t, D), axis=0)
        p2 = np.dot(X_t, self.y) + lmbda * np.sum(np.matmul(D_t, Z), axis=0)
        self.params = np.linalg.solve(p1, p2)
        return self.params

    def predict(self, x):
        X = self.poly_features.fit_transform(x)
        return np.dot(X, self.params)

# Set parameters
n = 200  # Number of samples
n_test = 1000  # Number of samples for testing fit
m = 5  # Number of stocks in the basket
w = np.array([1/m for i in range(1, m+1)])  # Weight of individual stock in basket
strike = 100  # Strike of basket
rf_rate = 0.0  # Risk-free rate (0 to easily simulate in the Bachelier model)
vol = 30  # Volatility in the model
cov = np.identity(m) * (vol ** 2)  # Covariance matrix governing stocks in basket
basket_vol = np.sqrt(np.dot(np.transpose(w), np.dot(cov, w)))
T = 0.2  # Time-to-maturity of basket option


Bach = Bachelier(rf_rate, vol)  # Choice of model
xpoints, xpoints_basket, ypoints, Z = Bachelier.simulate_basket(n, m, 50, 150, strike, T, w, cov)  # Simulate needed data
xpoints_test, xpoints_basket_test, ypoints_test, Z_test = Bachelier.simulate_basket(n_test, m, 50, 150, strike, T, w, cov)  # Simulate test-data


lmbda = np.dot(np.transpose(ypoints), ypoints) / np.dot(np.transpose(Z[0, :, :]), Z[0, :, :])  # Lambda for regularization, weighted by variation of term
print("lambda:\n{}".format(lmbda))


# Black-Scholes simulated data for regression of call option
n = 200
rf_rate = 0.02  # Risk-free rate
vol = 0.30  # Volatility in the model

BS = BlackScholes(rf_rate, vol)
xpoints_bs, ypoints_bs, D = BS.simulate_data(n, 50, 150, strike, T)

lmbda_BS = np.mean(ypoints ** 2) / np.mean(D ** 2)

p = 4  # Highest number of polynomial for polynomial regression
# Polynomial regression objects for estimating Black-Scholes price in 1D
poly_BS = PolyReg(xpoints_bs, ypoints_bs, p)
poly_BS.fit()

poly_BS_l2reg = PolyReg(xpoints_bs, ypoints_bs, p)
poly_BS_l2reg.fit(lmbda_BS)

poly_BS_dreg = PolyReg(xpoints_bs, ypoints_bs, p)
poly_BS_dreg.fit_differential(D, lmbda_BS)

# Polynomial regression objects for estimating Bachelier price in multi-dim
poly = PolyReg(xpoints, ypoints, p)
poly.fit()

poly_l2reg = PolyReg(xpoints, ypoints, p)
poly_l2reg.fit(lmbda)

poly_dreg = PolyReg(xpoints, ypoints, p)
poly_dreg.fit_differential(Z, lmbda)

#########_______.....Plotting....._______#########

# BLACK-SCHOLES PLOT
# Classic polynomial regression
plt.subplot(3, 2, 1)
plt.scatter(xpoints_bs, ypoints_bs, alpha=0.1)
plt.plot(xpoints_bs, poly_BS.predict(xpoints_bs), color='red')
plt.plot(xpoints_bs, BS.price(xpoints_bs, strike, T), color='green', linestyle='dotted')

# Polynomial regression with L2-regularization
plt.subplot(3, 2, 3)
plt.scatter(xpoints_bs, ypoints_bs, alpha=0.1)
plt.plot(xpoints_bs, poly_BS_l2reg.predict(xpoints_bs), color='red')
plt.plot(xpoints_bs, BS.price(xpoints_bs, strike, T), color='green', linestyle='dotted')

# Polynomial regression with differential regularization
plt.subplot(3, 2, 5)
plt.scatter(xpoints_bs, ypoints_bs, alpha=0.1)
plt.plot(xpoints_bs, poly_BS_dreg.predict(xpoints_bs), color='red')
plt.plot(xpoints_bs, BS.price(xpoints_bs, strike, T), color='green', linestyle='dotted')


# PLOT FOR BASKET
# Classic polynomial regression
plt.subplot(3, 2, 2)
plt.scatter(xpoints_basket, ypoints, alpha=0.1)
plt.scatter(xpoints_basket_test, poly.predict(xpoints_test), color='red', s=1, alpha=0.4)
plt.scatter(xpoints_basket_test, Bach.price(xpoints_basket_test, strike, T, basket_vol), color='green', s=1)

# Polynomial regression with L2-regularization
plt.subplot(3, 2, 4)
plt.scatter(xpoints_basket, ypoints, alpha=0.1)
plt.scatter(xpoints_basket_test, poly_l2reg.predict(xpoints_test), color='red', s=1, alpha=0.4)
plt.scatter(xpoints_basket_test, Bach.price(xpoints_basket_test, strike, T, basket_vol), color='green', s=1)

# Polynomial regression with differential regularization
plt.subplot(3, 2, 6)
plt.scatter(xpoints_basket, ypoints, alpha=0.1)
plt.scatter(xpoints_basket_test, poly_dreg.predict(xpoints_test), color='red', s=1, alpha=0.4)
plt.scatter(xpoints_basket_test, Bach.price(xpoints_basket_test, strike, T, basket_vol), color='green', s=1)

plt.show()
