import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from models import Bachelier_helper, BlackScholes_helper, Bachelier_eulerScheme, BlackScholes_eulerScheme, simulate_data
from options import Call, Call_basket, Call_geometric

class PolyReg:
    def __init__(self, x, y, p):
        self.x = x
        self.y = y
        self.p = p
        self.poly_features = PolynomialFeatures(degree=p)
        self.X = self.poly_features.fit_transform(x)
        self.params = None

    def predict(self, x):
        X = self.poly_features.fit_transform(x)
        return np.dot(X, self.params)

    def differential(self, x):
        X = self.poly_features.fit_transform(x)
        powers = self.poly_features.powers_#.reshape(-1, 1)
        D = np.transpose(np.transpose(powers)[:, :, np.newaxis] * np.transpose(X), axes=(0, 2, 1)) / \
            np.transpose(x[:, :, np.newaxis], axes=(1, 0, 2))

        return np.sum(np.dot(D, self.params), axis=0).reshape(-1, 1)
        #return np.dot(D, self.params).reshape(-1, 1)

    def integrated(self, x):
        X = self.poly_features.fit_transform(x)
        powers = self.poly_features.powers_ + 1
        D = np.transpose(np.transpose(X) / np.transpose(powers)[:, :, np.newaxis], axes=(0, 2, 1)) * \
            np.transpose(x[:, :, np.newaxis], axes=(1, 0, 2))

        return np.dot(D, self.params).reshape(-1, 1)

    def fit(self, lmbda=0):
        n = np.shape(self.X)[1]
        X_t = np.transpose(self.X)
        self.params = np.dot(np.linalg.solve(np.dot(X_t, self.X) + np.identity(n) * lmbda, X_t), self.y)
        return self.params

    def fit_differential(self, Z, lmbda=None):
        if lmbda is None:
            lmbda = np.dot(np.transpose(self.y), self.y) / np.dot(np.transpose(Z[0, :, :]), Z[0, :, :])

        X_t = np.transpose(self.X)
        powers = self.poly_features.powers_
        D = np.transpose(np.transpose(powers)[:, :, np.newaxis] * np.transpose(self.X), axes=(0, 2, 1)) / \
            np.transpose(self.x[:, :, np.newaxis], axes=(1, 0, 2))
        D_t = np.transpose(D, axes=(0, 2, 1))
        Z = np.transpose(Z[:, :, np.newaxis], axes=(1, 0, 2))
        p1 = np.dot(X_t, self.X) + lmbda * np.sum(np.matmul(D_t, D), axis=0)
        p2 = np.dot(X_t, self.y) + lmbda * np.sum(np.matmul(D_t, Z), axis=0)
        self.params = np.linalg.solve(p1, p2)
        return self.params

'''
# Set parameters
n = 1000  # Number of samples
n_test = 1000  # Number of samples for testing fit
m = 5  # Number of stocks in the basket
w = np.array([1/m for _ in range(m)])  # Weight of individual stock in basket
strike = 100  # Strike of basket
rf_rate = 0.0  # Risk-free rate (0 to easily simulate in the Bachelier_helper model)
vol = 50  # Volatility in the model
cov = np.identity(m) * (vol ** 2)  # Covariance matrix governing stocks in basket
basket_vol = np.sqrt(np.dot(np.transpose(w), np.dot(cov, w)))
T = 0.3  # Time-to-maturity of basket call
spot_rng = [50, 150]
test_rng = [50, 150]

basket = Call_basket(strike, T, w)

Bach_helper = Bachelier_helper(rf_rate, vol)  # Helper for analytic price etc.
Bach_model = Bachelier_eulerScheme(rf_rate, cov)

x, y, Z = simulate_data([n, m], spot_rng, basket, Bach_model, seed=1)
x_basket = np.reshape(x @ w.T, (-1, 1))

x_test, y_test, Z_test = simulate_data([n_test, m], test_rng, basket, Bach_model)
x_basket_test = np.reshape(x_test @ w.T, (-1, 1))

Z_perm = np.transpose(Z[:, :, np.newaxis], axes=(1, 0, 2))
lmbda = np.dot(np.transpose(y), y) / np.dot(np.transpose(Z_perm[0, :, :]), Z_perm[0, :, :])  # Lambda for regularization, weighted by variation of term
print("lambda: {}".format(lmbda))


# Black-Scholes simulated data for regression of options
vol = 0.20  # Volatility in the model (way lower in the Black-Scholes model)
rf_rate = 0.02  # Risk-free rate (0 to easily simulate in the Bachelier_helper model)
call = Call(strike, T)
'''
'''
T = 1
dt = 1/252
end = T * 252
T_geometric_avg = np.mean(np.arange(1, end+1) * dt)

var_geometric_avg = 0
for i in range(1, end+1):
    var_geometric_avg += (2*i - 1) * (T - dt * (i - 1))
var_geometric_avg = vol**2 / (end**2 * T_geometric_avg) * var_geometric_avg
delta = 0.5 * (vol**2 - var_geometric_avg)
'''
'''
BS_helper = BlackScholes_helper(rf_rate, vol)  # Helper for analytic price etc.
BS_model = BlackScholes_eulerScheme(rf_rate, vol**2)
x_bs, y_bs, Z_bs = simulate_data(n, spot_rng, call, BS_model, seed=4)

x_bs_test, y_bs_test, Z_bs_test = simulate_data(n_test, test_rng, call, BS_model)

order = np.argsort(x_bs_test, axis=0).flatten()
x_bs_test = x_bs_test[order]
y_bs_test = y_bs_test[order]
Z_bs_test = Z_bs_test[order]

lmbda_BS = np.mean(y_bs ** 2) / np.mean(Z_bs ** 2)

p = 5  # Highest number of polynomial for polynomial regression

# Polynomial regression objects for estimating Black-Scholes price in 1D
poly_BS = PolyReg(x_bs, y_bs, p)
poly_BS.fit()

poly_BS_l2reg = PolyReg(x_bs, y_bs, p)
poly_BS_l2reg.fit(lmbda_BS)

poly_BS_dreg = PolyReg(x_bs, y_bs, p)
poly_BS_dreg.fit_differential(Z_bs, lmbda_BS)

# Polynomial regression objects for estimating Bachelier_helper price in multi-dim
poly = PolyReg(x, y, p)
poly.fit()

poly_l2reg = PolyReg(x, y, p)
poly_l2reg.fit(lmbda)

poly_dreg = PolyReg(x, y, p)
poly_dreg.fit_differential(Z, lmbda)

'''
#########--------.....Plotting.....--------#########
'''
## Plot of fitted delta, and resulting pricing function
plt.rc('font', size=8)
plt.figure(figsize=(10, 2))
plt.subplot(1, 2, 1)
plt.plot(x_bs_test, poly_BS.integrated(x_bs_test), color='red', label='Estimated')
plt.plot(x_bs_test, BS_helper.call_price(x_bs_test, strike, T), color='green', linestyle='dotted', label='Analytic')
plt.legend(frameon=False, loc=2, markerfirst=False, fontsize=7, markerscale=2)

plt.subplot(1, 2, 2)
plt.scatter(x_bs, Z_bs, alpha=0.1, s=20)
plt.plot(x_bs_test, poly_BS.predict(x_bs_test), color='red', label='Estimated')
plt.plot(x_bs_test, BS_helper.call_delta(x_bs_test, strike, T), color='green', linestyle='dotted', label='Analytic')
plt.legend(frameon=False, loc=2, markerfirst=False, fontsize=7, markerscale=2)

plt.savefig('Graphs/BS_deltaregonly.png', dpi=1000, bbox_inches='tight')
plt.show()
'''

'''
# Plot of fitted pricing function with resulting delta
plt.rc('font', size=8)
plt.figure(figsize=(10, 2))
plt.subplot(1, 2, 1)
plt.scatter(x_bs, y_bs, alpha=0.1, s=20)
plt.plot(x_bs_test, poly_BS.predict(x_bs_test), color='red', label='Estimated')
plt.plot(x_bs_test, BS_helper.call_price(x_bs_test, strike, T), color='green', linestyle='dotted', label='Analytic')
plt.legend(frameon=False, loc=2, markerfirst=False, fontsize=7, markerscale=2)

plt.subplot(1, 2, 2)
plt.plot(x_bs_test, poly_BS.differential(x_bs_test), color='red', label='Estimated')
plt.plot(x_bs_test, BS_helper.call_delta(x_bs_test, strike, T), color='green', linestyle='dotted', label='Analytic')
plt.legend(frameon=False, loc=2, markerfirst=False, fontsize=7, markerscale=2)

#plt.savefig('Graphs/BS_price_delta.png', dpi=1000, bbox_inches='tight')
plt.show()
'''

'''
# Plot of pricing function and delta with differential regression
plt.rc('font', size=8)
plt.figure(figsize=(10, 2))
plt.subplot(1, 2, 1)
plt.scatter(x_bs, y_bs, alpha=0.1, s=20)
plt.plot(x_bs_test, poly_BS_dreg.predict(x_bs_test), color='red', label='Estimated')
plt.plot(x_bs_test, BS_helper.call_price(x_bs_test, strike, T), color='green', linestyle='dotted', label='Analytic')
plt.legend(frameon=False, loc=2, markerfirst=False, fontsize=7, markerscale=2)

plt.subplot(1, 2, 2)
plt.scatter(x_bs, Z_bs, alpha=0.1, s=20)
plt.plot(x_bs_test, poly_BS_dreg.differential(x_bs_test), color='red', label='Estimated')
plt.plot(x_bs_test, BS_helper.call_delta(x_bs_test, strike, T), color='green', linestyle='dotted', label='Analytic')
plt.legend(frameon=False, loc=2, markerfirst=False, fontsize=7, markerscale=2)

plt.savefig('Graphs/BS_price_delta_diff.png', dpi=1000, bbox_inches='tight')
plt.show()
'''

'''
# BLACK-SCHOLES PLOT
# Classic polynomial regression
plt.subplot(2, 1, 1)
plt.scatter(x_bs, y_bs, alpha=0.1, s=20)
plt.plot(x_bs_test, poly_BS.predict(x_bs_test), color='red')
plt.plot(x_bs_test, BS_helper.call_price(x_bs_test, strike, T), color='green', linestyle='dotted')
#plt.plot(x_bs, BS_helper.call_price(x_bs, strike, T_geometric_avg, vol=np.sqrt(var_geometric_avg), rf_rate=rf_rate-delta), color='green', linestyle='dotted')

plt.subplot(2, 1, 2)
#plt.scatter(x_bs, Z_bs, alpha=0.1, s=20)
plt.plot(x_bs_test, poly_BS.differential(x_bs_test), color='red')
plt.plot(x_bs_test, BS_helper.call_delta(x_bs_test, strike, T), color='green', linestyle='dotted')
#plt.plot(x_bs, BS_helper.call_delta(x_bs, strike, T_geometric_avg), color='green', linestyle='dotted')

# Polynomial regression with L2-regularization
plt.subplot(3, 2, 3)
plt.scatter(x_bs, y_bs, alpha=0.1, s=20)
plt.plot(x_bs_test, poly_BS_l2reg.predict(x_bs_test), color='red')
plt.plot(x_bs_test, BS_helper.call_price(x_bs_test, strike, T), color='green', linestyle='dotted')
#plt.plot(x_bs, BS_helper.call_price(x_bs, strike, T_geometric_avg, vol=np.sqrt(var_geometric_avg), rf_rate=rf_rate-delta), color='green', linestyle='dotted')

plt.subplot(3, 2, 4)
plt.scatter(x_bs, Z_bs, alpha=0.1, s=20)
plt.plot(x_bs_test, poly_BS_l2reg.differential(x_bs_test), color='red')
plt.plot(x_bs_test, BS_helper.call_delta(x_bs_test, strike, T), color='green', linestyle='dotted')
#plt.plot(x_bs, BS_helper.call_delta(x_bs, strike, T_geometric_avg), color='green', linestyle='dotted')

# Polynomial regression with differential regularization
plt.subplot(3, 2, 5)
plt.scatter(x_bs, y_bs, alpha=0.1, s=20)
plt.plot(x_bs_test, poly_BS_dreg.predict(x_bs_test), color='red')
plt.plot(x_bs_test, BS_helper.call_price(x_bs_test, strike, T), color='green', linestyle='dotted')
#plt.plot(x_bs, BS_helper.call_price(x_bs, strike, T_geometric_avg, vol=np.sqrt(var_geometric_avg), rf_rate=rf_rate-delta), color='green', linestyle='dotted')

plt.subplot(3, 2, 6)
plt.scatter(x_bs, Z_bs, alpha=0.1, s=20)
plt.plot(x_bs_test, poly_BS_dreg.differential(x_bs_test), color='red')
plt.plot(x_bs_test, BS_helper.call_delta(x_bs_test, strike, T), color='green', linestyle='dotted')
#plt.plot(x_bs, BS_helper.call_delta(x_bs, strike, T_geometric_avg), color='green', linestyle='dotted')

plt.show()
'''

'''
# Plot for basket call pricing function and resulting delta
plt.rc('font', size=8)
plt.figure(figsize=(9, 4))
plt.subplot(2, 2, 1)
plt.scatter(x_basket, y, alpha=0.1, s=20)
plt.plot(x_basket_test, poly.predict(x_test), color='red', alpha=0.4, marker="x", ls='none', ms=3, mew=0.5, label='Estimated')
plt.plot(x_basket_test, Bach_helper.call_price(x_basket_test, strike, T, basket_vol), color='green', linestyle='none', marker='x', ms=3, mew=0.5, alpha=0.4, label='Analytic')
plt.ylim(-5, 49)
plt.legend(frameon=False, loc=2, markerfirst=False, fontsize=7, markerscale=2)

plt.subplot(2, 2, 2)
plt.plot(x_basket_test, poly.differential(x_test), color='red', alpha=0.4, marker="x", ls='none', ms=3, mew=0.5, label='Estimated')
plt.plot(x_basket_test, Bach_helper.call_delta(x_basket_test, strike, T, basket_vol), color='green', linestyle='none', marker='x', ms=3, mew=0.5, alpha=0.4, label='Analytic')
plt.ylim(-0.4, 1.9)
plt.legend(frameon=False, loc=2, markerfirst=False, fontsize=7, markerscale=2)

# Plot for fit of pricing function and delta using differential regression
plt.subplot(2, 2, 3)
plt.scatter(x_basket, y, alpha=0.1, s=20)
plt.plot(x_basket_test, poly_dreg.predict(x_test), color='red', alpha=0.4, marker="x", ls='none', ms=3, mew=0.5, label='Estimated')
plt.plot(x_basket_test, Bach_helper.call_price(x_basket_test, strike, T, basket_vol), color='green', linestyle='none', marker='x', ms=3, mew=0.5, alpha=0.4, label='Analytic')
plt.ylim(-5, 49)
plt.legend(frameon=False, loc=2, markerfirst=False, fontsize=7, markerscale=2)

plt.subplot(2, 2, 4)
plt.scatter(x_basket, np.sum(Z, axis=1), alpha=0.1, s=20)
plt.plot(x_basket_test, poly_dreg.differential(x_test), color='red', alpha=0.4, marker="x", ls='none', ms=3, mew=0.5, label='Estimated')
plt.plot(x_basket_test, Bach_helper.call_delta(x_basket_test, strike, T, basket_vol), color='green', linestyle='none', marker='x', ms=3, mew=0.5, alpha=0.4, label='Analytic')
plt.ylim(-0.4, 1.9)
plt.legend(frameon=False, loc=2, markerfirst=False, fontsize=7, markerscale=2)

plt.savefig('Graphs/Bachelier_price_delta.png', dpi=1000, bbox_inches='tight')
plt.show()
'''

'''
# PLOT FOR BASKET
# Classic polynomial regression
plt.subplot(3, 1, 1)
plt.scatter(x_basket, y, alpha=0.1, s=20)
plt.plot(x_basket_test, Bach_helper.call_price(x_basket_test, strike, T, basket_vol), color='green', linestyle='none', marker='x', ms=3, mew=0.5, alpha=0.4)
plt.plot(x_basket_test, poly.predict(x_test), color='red', alpha=0.4, marker="x", ls='none', ms=3, mew=0.5)
plt.ylim(-5, 50)

# Polynomial regression with L2-regularization
plt.subplot(3, 1, 2)
plt.scatter(x_basket, y, alpha=0.1, s=20)
plt.plot(x_basket_test, Bach_helper.call_price(x_basket_test, strike, T, basket_vol), color='green', linestyle='none', marker='x', ms=3, mew=0.5, alpha=0.4)
plt.plot(x_basket_test, poly_l2reg.predict(x_test), color='red', alpha=0.4, marker="x", ls='none', ms=3, mew=0.5)
plt.ylim(-5, 50)

# Polynomial regression with differential regularization
plt.subplot(3, 1, 3)
plt.scatter(x_basket, y, alpha=0.1, s=20)
plt.plot(x_basket_test, Bach_helper.call_price(x_basket_test, strike, T, basket_vol), color='green', linestyle='none', marker='x', ms=3, mew=0.5, alpha=0.4)
plt.plot(x_basket_test, poly_dreg.predict(x_test), color='red', alpha=0.4, marker="x", ls='none', ms=3, mew=0.5)
plt.ylim(-5, 50)

# plt.savefig('foo.png', dpi=1000, bbox_inches='tight')

plt.show()

'''


