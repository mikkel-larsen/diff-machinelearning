import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.special import binom

from models import Bachelier
from models import Basket


n = 500
m = 5
w = np.array([1/m for i in range(1, m+1)])
strike = 100
rf_rate = 0.0
vol = 30
cov = np.identity(m) * (vol ** 2)
T = 0.2

Bach = Bachelier(rf_rate, vol)
basket = Basket(m, w, strike)

basket.make_basket_uniform(50, 150, n)

sim = np.array(Bach.simulate_basket_endpoint(basket, cov, T))
# print("sim: {}".format(sim))
sim_basket = np.dot(sim, w).reshape((-1, 1))
# print("sim cumulative: {}".format(sim_basket))
ypoints = np.array(np.maximum(sim_basket - strike, 0))
# print("y: {}".format(ypoints))


def poly(x, params, p):
    X = PolynomialFeatures(degree=(p)).fit_transform(x)
    return np.dot(X, params)

def design_matrix(x, p):
    return PolynomialFeatures(degree=p).fit_transform(x)

def fit_poly(x, y, p, lmbda=0):
    X = design_matrix(x, p)
    X_t = np.transpose(X)
    n = np.shape(X)[1]
    return np.dot(np.linalg.solve(np.dot(X_t, X) + np.identity(n) * lmbda, X_t), y)

def fit_poly_diff(x, y, D, p, lmbda=None):
    if lmbda == None:
        lmbda = np.mean(y ** 2) / np.mean(D ** 2)

    poly_features = PolynomialFeatures(degree=p)
    X = poly_features.fit_transform(x)
    X_t = np.transpose(X)
    powers = poly_features.powers_
    D = np.transpose(np.transpose(powers)[:, :, np.newaxis] * np.transpose(X), axes=(0, 2, 1)) / np.transpose(x[:, :, np.newaxis], axes=(1, 0, 2))
    D_t = np.transpose(D, axes=(0, 2, 1))
    '''Y = design_matrix(x, p-1)
    Y = np.insert(Y, 0, np.zeros(n), 1)
    i = [y for y in range(p+1)]
    Y = np.multiply(Y, i)'''
    p1 = np.dot(X_t, X) + lmbda * np.sum(np.matmul(D_t, D), axis=0)
    p2 = np.dot(X_t, y) + lmbda * np.sum(np.matmul(D_t, Z), axis=0)
    return np.linalg.solve(p1, p2)


Z = np.dot(np.where(sim_basket > strike, 1, 0).reshape(-1, 1), w.reshape(1, -1))
Z = np.transpose(Z[:, :, np.newaxis], axes=(1, 0, 2))
print("Z shape:\n{}".format(np.shape(Z)))

print("Y^2 / Z^2: {}".format(np.dot(np.transpose(ypoints), ypoints) / np.dot(np.transpose(Z[0, :, :]), Z[0, :, :])))
print("sd(D) / (sd(D) + sd(C): {}".format(np.std(Z) / (np.std(Z) + np.std(ypoints))))

lmbda = np.dot(np.transpose(ypoints), ypoints) / np.dot(np.transpose(Z[0, :, :]), Z[0, :, :])
print("lambda:\n{}".format(lmbda))
p = 4

xpoints_basket = np.dot(basket.spots, w)

n = 1000
basket_test = Basket(m, w, strike)
basket_test.make_basket_uniform(50, 150, n)
xpoints_basket_test = np.dot(basket_test.spots, w)

sim_test = np.array(Bach.simulate_basket_endpoint(basket_test, cov, T))
sim_basket_test = np.dot(sim_test, w).reshape((-1, 1))
ypoints_test = np.array(np.maximum(sim_basket_test - strike, 0))

plt.subplot(3, 1, 1)
plt.scatter(xpoints_basket, ypoints, alpha=0.1)
plt.scatter(xpoints_basket_test, poly(basket_test.spots, fit_poly(basket.spots, ypoints, p), p), color='red', s=1)
plt.scatter(xpoints_basket_test, Bach.price_basket(basket_test, cov, T), color='green', s=1)

plt.subplot(3, 1, 2)
plt.scatter(xpoints_basket, ypoints, alpha=0.1)
plt.scatter(xpoints_basket_test, poly(basket_test.spots, fit_poly(basket.spots, ypoints, p, lmbda*10), p), color='red', s=1)
plt.scatter(xpoints_basket_test, Bach.price_basket(basket_test, cov, T), color='green', s=1)

plt.subplot(3, 1, 3)
plt.scatter(xpoints_basket, ypoints, alpha=0.1)
plt.scatter(xpoints_basket_test, poly(basket_test.spots, fit_poly_diff(basket.spots, ypoints, Z, p, lmbda), p), color='red', s=1)
plt.scatter(xpoints_basket_test, Bach.price_basket(basket_test, cov, T), color='green', s=1)

plt.show()
