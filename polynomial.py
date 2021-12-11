import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from simulate_data import BS
from simulate_data import transform

n = 200
strike = 100
rf_rate = 0.02
vol = 0.15
T = 0.3

model = BS(rf_rate, vol, T)
xpoints = transform(np.linspace(20, 180, n), 100)
sim = model.BS_simulate_endpoint(n, xpoints)
ypoints = np.maximum(sim - strike, 0)

def true_BS(spot, rf_rate, vol, T, strike):
    d1 = (np.log(spot / strike) + (rf_rate + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return spot * norm.cdf(d1) - strike * np.exp(-rf_rate * T) * norm.cdf(d2)

def poly(x, params):
    ret = 0
    for p in range(len(params)):
        ret += params[p] * x ** p
    return ret

def design_matrix(x, p):
    X = np.array([x ** i for i in range(p+1)])
    return np.transpose(X)

def fit_poly(x, y, p, lmbda=0):
    X = design_matrix(x, p)
    X_t = np.transpose(X)
    p = len(X_t)
    return np.dot(np.linalg.solve(np.dot(X_t, X) + np.identity(p) * lmbda, X_t), y)

def fit_poly_diff(x, c, D, p, lmbda=None):
    if lmbda == None:
        lmbda = np.mean(c ** 2) / np.mean(D ** 2)
    X = design_matrix(x, p)
    X_t = np.transpose(X)
    n = len(x)
    Y = design_matrix(xpoints, p-1)
    Y = np.insert(Y, 0, np.zeros(n), 1)
    i = [y for y in range(p+1)]
    Y = np.multiply(Y, i)
    Y_t = np.transpose(Y)
    return np.linalg.solve(np.dot(X_t, X) + lmbda * np.dot(Y_t, Y), np.dot(X_t, c) + lmbda * np.dot(Y_t, D))
    #return np.linalg.solve(w * np.dot(X_t, X) + (1 - w) * np.dot(Y_t, Y), w * np.dot(X_t, c) + (1 - w) * np.dot(Y_t, D))


D = [1 if x > strike else 0 for x in sim] * sim / xpoints

print("Y^2 / Z^2: {}".format(np.mean(ypoints ** 2) / np.mean(D ** 2)))
print("sd(D) / (sd(D) + sd(C): {}".format(np.std(D) / (np.std(D) + np.std(ypoints))))


lmbda = np.mean(ypoints ** 2) / np.mean(D ** 2)

plt.subplot(2,1,1)
plt.scatter(xpoints, ypoints, alpha=0.1)
plt.plot(xpoints, true_BS(xpoints, rf_rate, vol, T, strike), linestyle='dotted')
plt.plot(xpoints, poly(xpoints, fit_poly_diff(xpoints, ypoints, D, 7)), color='red', linestyle='solid')

plt.subplot(2,1,2)
plt.scatter(xpoints, ypoints, alpha=0.1)
plt.plot(xpoints, true_BS(xpoints, rf_rate, vol, T, strike), linestyle='dotted')
plt.plot(xpoints, poly(xpoints, fit_poly(xpoints, ypoints, 7)), color='red', linestyle='solid')

plt.show()