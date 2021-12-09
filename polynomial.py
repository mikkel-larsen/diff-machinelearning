import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from scipy.stats import norm

from simulate_data import BS
from simulate_data import transform

n = 10000
strike = 100
rf_rate = 0.02
vol = 0.15
T = 0.5

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

def fit_parameters(X, y):
    X_t = np.transpose(X)
    return np.dot(np.linalg.solve(np.dot(X_t, X), X_t), y)


fitted_params = fit_parameters(design_matrix(xpoints, 5), ypoints)
print("Fitted parameters ({0}): {1}".format(len(fitted_params), fitted_params))


plt.scatter(xpoints, ypoints, alpha=0.1)
plt.plot(xpoints, true_BS(xpoints, rf_rate, vol, T, strike), linestyle='dotted')
plt.plot(xpoints, poly(xpoints, fitted_params), color='red')
plt.show()
