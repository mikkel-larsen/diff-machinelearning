import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.special import binom

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
m = 5  # Number of stocks in the basket
w = np.array([1/m for i in range(1, m+1)])  # Weight of individual stock in basket
strike = 100  # Strike of basket
rf_rate = 0.0  # Risk-free rate (0 to easily simulate in the Bachelier model)
vol = 30  # Volatility in the model
cov = np.identity(m) * (vol ** 2)  # Covariance matrix governing stocks in basket
T = 0.2  # Time-to-maturity of basket option

# Simulate data with Bachelier for regression of basket option
Bach = Bachelier(rf_rate, vol)  # Choice of model
basket = Basket(m, w, strike)  # Initialize basket (for training) object with number of stocks, weights, and strike

basket.make_basket_uniform(50, 150, n)  # Randomly pick spots (uniformly) between min and max, with n samples

sim = np.array(Bach.simulate_basket_endpoint(basket, cov, T))  # Simulate path for each sample of each stock in basket
sim_basket = np.dot(sim, w).reshape((-1, 1))  # Weigh stock price in basket using basket weights
ypoints = np.array(np.maximum(sim_basket - strike, 0))  # Simulated payoffs (european call option)
Z = np.dot(np.where(sim_basket > strike, 1, 0).reshape(-1, 1), w.reshape(1, -1))  # Pathwise differentials
Z = np.transpose(Z[:, :, np.newaxis], axes=(1, 0, 2))  # transpose so dimensions fit

print("Y^2 / Z^2: {}".format(np.dot(np.transpose(ypoints), ypoints) / np.dot(np.transpose(Z[0, :, :]), Z[0, :, :])))
print("sd(D) / (sd(D) + sd(C): {}".format(np.std(Z) / (np.std(Z) + np.std(ypoints))))

xpoints_basket = np.dot(basket.spots, w)  # Basket weighted spot

lmbda = np.dot(np.transpose(ypoints), ypoints) / np.dot(np.transpose(Z[0, :, :]), Z[0, :, :])  # Lambda for regularization, weighted by variation of term
print("lambda:\n{}".format(lmbda))

# Basket for testing --------------------------------------------------
n = 1000  # Number of samples for testing fit
basket_test = Basket(m, w, strike)  # New basket object for testing fit
basket_test.make_basket_uniform(50, 150, n)
xpoints_basket_test = np.dot(basket_test.spots, w)


# Black-Scholes simulated data for regression of call option
rf_rate = 0.02  # Risk-free rate
vol = 0.30  # Volatility in the model

BS = BlackScholes(rf_rate, vol)

xpoints_bs = np.sort(random.uniform(50, 150, n)).reshape(-1, 1)
sim_bs = BS.simulate_endpoint(xpoints_bs, T)
ypoints_bs = np.maximum(0, sim_bs - strike)

p = 4  # Highest number of polynomial for polynomial regression

D = np.where(sim_bs > strike, 1, 0) * sim_bs / xpoints_bs
lmbda_BS = np.mean(ypoints ** 2) / np.mean(D ** 2)

# Polynomial regression objects for estimating Black-Scholes price in 1D
poly_BS = PolyReg(xpoints_bs, ypoints_bs, p)
poly_BS.fit()

poly_BS_l2reg = PolyReg(xpoints_bs, ypoints_bs, p)
poly_BS_l2reg.fit(lmbda_BS)

poly_BS_dreg = PolyReg(xpoints_bs, ypoints_bs, p)
poly_BS_dreg.fit_differential(D, lmbda_BS)

# Polynomial regression objects for estimating Bachelier price in multi-dim
poly = PolyReg(basket.spots, ypoints, p)
poly.fit()

poly_l2reg = PolyReg(basket.spots, ypoints, p)
poly_l2reg.fit(lmbda)

poly_dreg = PolyReg(basket.spots, ypoints, p)
poly_dreg.fit_differential(Z, lmbda)

#########_______.....Plotting....._______#########

# BLACK-SCHOLES PLOT
# Classic polynomial regression
plt.subplot(3, 2, 1)
plt.scatter(xpoints_bs, ypoints_bs, alpha=0.1)
plt.plot(xpoints_bs, poly_BS.predict(xpoints_bs))
plt.plot(xpoints_bs, BS.price(xpoints_bs, strike, T), color='green', linestyle='dotted')

# Polynomial regression with L2-regularization
plt.subplot(3, 2, 3)
plt.scatter(xpoints_bs, ypoints_bs, alpha=0.1)
plt.plot(xpoints_bs, poly_BS_l2reg.predict(xpoints_bs))
plt.plot(xpoints_bs, BS.price(xpoints_bs, strike, T), color='green', linestyle='dotted')

# Polynomial regression with differential regularization
plt.subplot(3, 2, 5)
plt.scatter(xpoints_bs, ypoints_bs, alpha=0.1)
plt.plot(xpoints_bs, poly_BS_dreg.predict(xpoints_bs))
plt.plot(xpoints_bs, BS.price(xpoints_bs, strike, T), color='green', linestyle='dotted')


# PLOT FOR BASKET
# Classic polynomial regression
plt.subplot(3, 2, 2)
plt.scatter(xpoints_basket, ypoints, alpha=0.1)
plt.scatter(xpoints_basket_test, poly.predict(basket_test.spots), color='red', s=1)
plt.scatter(xpoints_basket_test, Bach.price_basket(basket_test, cov, T), color='green', s=1)

# Polynomial regression with L2-regularization
plt.subplot(3, 2, 4)
plt.scatter(xpoints_basket, ypoints, alpha=0.1)
plt.scatter(xpoints_basket_test, poly_l2reg.predict(basket_test.spots), color='red', s=1)
plt.scatter(xpoints_basket_test, Bach.price_basket(basket_test, cov, T), color='green', s=1)

# Polynomial regression with differential regularization
plt.subplot(3, 2, 6)
plt.scatter(xpoints_basket, ypoints, alpha=0.1)
plt.scatter(xpoints_basket_test, poly_dreg.predict(basket_test.spots), color='red', s=1)
plt.scatter(xpoints_basket_test, Bach.price_basket(basket_test, cov, T), color='green', s=1)

plt.show()
