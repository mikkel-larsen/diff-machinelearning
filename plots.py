import keras.models
import numpy as np
import sklearn.decomposition
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mtick
from sklearn.preprocessing import StandardScaler
import time

from sklearn import decomposition

from polynomial import PolyReg
from models import Bachelier_helper, BlackScholes_helper, Bachelier_eulerScheme, BlackScholes_eulerScheme, simulate_data
from options import Call, Call_basket, Call_geometric, Spread, Linear, Digital
from ANN import NeuralNetwork, create_and_fit_diffANN, create_and_fit_ANN
from PCA import PCA

########################################################################################
############################### POLYNOMIAL REGRESSION ##################################
########################################################################################

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

vol_bs = 0.20  # Volatility in the model (way lower in the Black-Scholes model)
rf_rate_bs = 0.02  # Risk-free rate (0 to easily simulate in the Bachelier_helper model)

call = Call(strike, T)
basket = Call_basket(strike, T, w)

Bach_helper = Bachelier_helper(rf_rate, vol)  # Helper for analytic price etc.
Bach_model = Bachelier_eulerScheme(rf_rate, cov)

x, y, Z = simulate_data([n, m], spot_rng, basket, Bach_model, seed=1)
x_basket = np.reshape(x @ w.T, (-1, 1))

x_test, y_test, Z_test = simulate_data([n_test, m], test_rng, basket, Bach_model)
x_basket_test = np.reshape(x_test @ w.T, (-1, 1))
x_basket_linspace = np.linspace(min(x_basket_test), max(x_basket_test), 100)

Z_perm = np.transpose(Z[:, :, np.newaxis], axes=(1, 0, 2))
lmbda = np.dot(np.transpose(y), y) / np.dot(np.transpose(Z_perm[0, :, :]), Z_perm[0, :, :])  # Lambda for regularization, weighted by variation of term
print("lambda: {}".format(lmbda))

BS_helper = BlackScholes_helper(rf_rate_bs, vol_bs)  # Helper for analytic price etc.
BS_model = BlackScholes_eulerScheme(rf_rate_bs, vol_bs**2)
x_bs, y_bs, Z_bs = simulate_data(n, spot_rng, call, BS_model, seed=2)

x_bs_test, y_bs_test, Z_bs_test = simulate_data(n_test, test_rng, call, BS_model)

order_bs = np.argsort(x_bs_test, axis=0).flatten()
x_bs_test = x_bs_test[order_bs]
y_bs_test = y_bs_test[order_bs]
Z_bs_test = Z_bs_test[order_bs]

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
poly_BS = PolyReg(x_bs, Z_bs, p)
poly_BS.fit()
plt.rc('font', size=8)
plt.figure(figsize=(9, 2))
plt.subplot(1, 2, 1)
plt.plot(x_bs_test, poly_BS.integrated(x_bs_test), color='red', label='Estimated')
plt.plot(x_bs_test, BS_helper.call_price(x_bs_test, strike, T), color='green', linestyle='dotted', label='True')
plt.legend(frameon=False, loc=2, markerfirst=False, fontsize=7)
plt.xlabel('Spot')
plt.ylabel('Price')

plt.subplot(1, 2, 2)
plt.scatter(x_bs, Z_bs, alpha=0.1, s=20)
plt.plot(x_bs_test, poly_BS.predict(x_bs_test), color='red', label='Estimated')
plt.plot(x_bs_test, BS_helper.call_delta(x_bs_test, strike, T), color='green', linestyle='dotted', label='True')
plt.legend(frameon=False, loc=2, markerfirst=False, fontsize=7)
plt.xlabel('Spot')
plt.ylabel('Delta')

# plt.savefig('Graphs/BS_deltaregonly.png', dpi=1000, bbox_inches='tight')
plt.show()
'''

'''
# Plot of fitted pricing function with resulting delta
plt.rc('font', size=8)
plt.figure(figsize=(9, 2))
plt.subplot(1, 2, 1)
plt.scatter(x_bs, y_bs, alpha=0.1, s=20)
plt.plot(x_bs_test, poly_BS.predict(x_bs_test), color='red', label='Estimated')
plt.plot(x_bs_test, BS_helper.call_price(x_bs_test, strike, T), color='green', linestyle='dotted', label='Analytic')
plt.legend(frameon=False, loc=2, markerfirst=False, fontsize=7)
plt.xlabel('Spot')
plt.ylabel('Price')

plt.subplot(1, 2, 2)
plt.plot(x_bs_test, poly_BS.differential(x_bs_test), color='red', label='Estimated')
plt.plot(x_bs_test, BS_helper.call_delta(x_bs_test, strike, T), color='green', linestyle='dotted', label='Analytic')
plt.legend(frameon=False, loc=2, markerfirst=False, fontsize=7)
plt.xlabel('Spot')
plt.ylabel('Delta')

# plt.savefig('Graphs/BS_price_delta.png', dpi=1000, bbox_inches='tight')
plt.show()
'''

'''
# Plot of pricing function and delta with differential regression
plt.rc('font', size=8)
plt.figure(figsize=(9, 2))
plt.subplot(1, 2, 1)
plt.scatter(x_bs, y_bs, alpha=0.1, s=20)
plt.plot(x_bs_test, poly_BS_dreg.predict(x_bs_test), color='red', label='Estimated')
plt.plot(x_bs_test, BS_helper.call_price(x_bs_test, strike, T), color='green', linestyle='dotted', label='Analytic')
plt.legend(frameon=False, loc=2, markerfirst=False, fontsize=7)
plt.xlabel('Spot')
plt.ylabel('Price')

plt.subplot(1, 2, 2)
plt.scatter(x_bs, Z_bs, alpha=0.1, s=20)
plt.plot(x_bs_test, poly_BS_dreg.differential(x_bs_test), color='red', label='Estimated')
plt.plot(x_bs_test, BS_helper.call_delta(x_bs_test, strike, T), color='green', linestyle='dotted', label='Analytic')
plt.legend(frameon=False, loc=2, markerfirst=False, fontsize=7)
plt.xlabel('Spot')
plt.ylabel('Delta')

# plt.savefig('Graphs/BS_price_delta_diff.png', dpi=1000, bbox_inches='tight')
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
est, = plt.plot(x_basket_test, poly.predict(x_test), color='red', marker="x", ls='none', ms=3, mew=0.5, label='Estimated')
plt.plot(x_basket_linspace, Bach_helper.call_price(x_basket_linspace, strike, T, basket_vol), color='green', label='True')
plt.ylim(-5, 49)
plt.legend(frameon=False, loc=2, markerfirst=False, fontsize=7, markerscale=1.2)
est.set_alpha(0.4)
plt.ylabel('Price')

plt.subplot(2, 2, 2)
est, = plt.plot(x_basket_test, poly.differential(x_test), color='red', marker="x", ls='none', ms=3, mew=0.5, label='Estimated')
plt.plot(x_basket_linspace, Bach_helper.call_delta(x_basket_linspace, strike, T, basket_vol), color='green', label='True')
plt.ylim(-0.4, 1.9)
plt.legend(frameon=False, loc=2, markerfirst=False, fontsize=7, markerscale=1.2)
est.set_alpha(0.4)
plt.ylabel('Delta')

# Plot for fit of pricing function and delta using differential regression
plt.subplot(2, 2, 3)
plt.scatter(x_basket, y, alpha=0.1, s=20)
est, = plt.plot(x_basket_test, poly_dreg.predict(x_test), color='red', marker="x", ls='none', ms=3, mew=0.5, label='Estimated')
plt.plot(x_basket_linspace, Bach_helper.call_price(x_basket_linspace, strike, T, basket_vol), color='green', label='True')
plt.ylim(-5, 49)
plt.legend(frameon=False, loc=2, markerfirst=False, fontsize=7, markerscale=1.2)
est.set_alpha(0.4)
plt.xlabel('Spot')
plt.ylabel('Price')

plt.subplot(2, 2, 4)
plt.scatter(x_basket, np.sum(Z, axis=1), alpha=0.1, s=20)
est, = plt.plot(x_basket_test, poly_dreg.differential(x_test), color='red', marker="x", ls='none', ms=3, mew=0.5, label='Estimated')
plt.plot(x_basket_linspace, Bach_helper.call_delta(x_basket_linspace, strike, T, basket_vol), color='green', label='True')
plt.ylim(-0.4, 1.9)
plt.legend(frameon=False, loc=2, markerfirst=False, fontsize=7, markerscale=1.2)
est.set_alpha(0.4)
plt.xlabel('Spot')
plt.ylabel('Delta')

# plt.savefig('Graphs/Bachelier_price_delta.png', dpi=1000, bbox_inches='tight')
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

########################################################################################
################################ Neural Networks #######################################
########################################################################################
'''
# Set parameters
n = 1000  # Number of samples
n_test = 1000  # Number of samples for testing fit
strike = 100  # Strike of basket
rf_rate = 0.0  # Risk-free rate
T = 0.3  # Time-to-maturity of basket call
spot_rng = [50, 150]

# ---------- BLACK-SCHOLES ----------
vol_bs = 0.2  # Volatility in the model

# ---------- BACHELIER --------------
vol_bach = 50  # Volatility in the model
m = 5
w = np.array([1/m for _ in range(m)])  # Weight of individual stock in basket
cov = np.identity(m) * (vol_bach ** 2)  # Covariance matrix governing stocks in basket
#np.random.seed(7)
#cov = np.random.uniform(-100, 100, (m, m))
#cov = cov.T @ cov
basket_vol = np.sqrt(np.dot(np.transpose(w), np.dot(cov, w)))

# Simulate datasets for training and testing
#option = Call(strike, T)
option = Call_basket(strike, T, w)
Bach_helper = Bachelier_helper(rf_rate, basket_vol)
BS_helper = BlackScholes_helper(rf_rate, vol_bs)

model = Bachelier_eulerScheme(rf_rate, cov)
#model = BlackScholes_eulerScheme(rf_rate, vol_bs**2)

x, y, D = simulate_data([n, m], spot_rng, option, model, seed=2)
b = np.reshape(x @ w.T, (-1, 1))
#x_linspace = np.linspace(min(x), max(x), 5000).reshape(-1, 1)

x_test, y_test, D_test = simulate_data([n_test, m], spot_rng, option, model)
b_test = np.reshape(x_test @ w.T, (-1, 1))
b_test_linspace = np.linspace(min(b_test), max(b_test), 100)


# Create, fit and time models
t0 = time.time()
model = create_and_fit_ANN(x, y)
t1 = time.time()
training_time = t1 - t0

t0 = time.time()
model_diff = create_and_fit_diffANN(x, y, D)
#model_diff_relu = create_and_fit_diffANN(x, y, D, activations='relu')
t1 = time.time()
training_time_diff = t1 - t0

# Predict and find gradients
yhat_diff, yhat_diff_grad = model_diff.predict(x_test)  # x_linspace
#yhat_diff_relu, yhat_diff_grad_relu = model_diff_relu.predict(x_linspace)

x_test = tf.convert_to_tensor(x_test)
with tf.GradientTape() as tape:
    tape.watch(x_test)
    yhat = model(x_test)

yhat_grad = tape.gradient(yhat, x_test)

# index_test = np.random.randint(0, n_test, 3)
# print("Predicted regression: \n {} \n Correct: \n {}".format(yhat[index_test], BS_helper.call_price(x_test[index_test], strike, T)))
# print("Predicted gradient: \n {} \n Correct: \n {}".format(yhat_grad[index_test], BS_helper.call_delta(x_test[index_test], strike, T)))

scaler = StandardScaler()
scaler.fit(x)
Z_perm = np.transpose(D[:, :, np.newaxis], axes=(1, 0, 2))
lmbda = np.dot(np.transpose(y), y) / np.dot(np.transpose(Z_perm[0, :, :]), Z_perm[0, :, :]) # Lambda for regularization, weighted by variation of term
w = (lmbda / (1 + lmbda))
print("Weight: {}".format(w))
print("Training time: \n {}".format(t1-t0))
print("Basket vol: {}".format(basket_vol))
'''

# ---------- Plot ------------------------
'''
# Plot of relu activation function delta
def softplus(x):
    return np.log(1 + np.exp(x))

def relu(x):
    return np.maximum(x, 0)

z = np.linspace(-3, 3, 1000)

plt.rc('font', size=8)
plt.figure(figsize=(9, 2))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
a0 = plt.subplot(gs[0])
a0.plot(z, relu(z), color='red', label='ReLU')
a0.plot(z, softplus(z), color='blue', label='Softplus')
a0.legend(frameon=False, markerfirst=False, fontsize=7)

a1 = plt.subplot(gs[1])
a1.scatter(x, D, alpha=0.1, s=20)
a1.plot(x_linspace, BS_helper.call_delta(x_linspace, strike, T), color='green', linestyle='dotted', label='True')
a1.plot(x_linspace, yhat_diff_grad_relu, color='red', label='ReLU')
a1.plot(x_linspace, yhat_diff_grad, color='blue', label='Softplus')
a1.legend(frameon=False, loc=2, markerfirst=False, fontsize=7)
plt.xlabel('Spot')
plt.ylabel('Delta')

# plt.savefig('Graphs/BS_reludelta.png', dpi=1000, bbox_inches='tight')
plt.show()
'''

'''
# Plot of bachelier pricing function and the resulting delta using regular ANN
plt.rc('font', size=8)
plt.figure(figsize=(9, 2))
plt.subplot(1, 2, 1)
plt.scatter(b, y, alpha=0.1, s=20)
est, = plt.plot(b_test, yhat, color='red', marker="x", ls='none', ms=3, mew=0.5, label='Estimated')
plt.plot(b_test_linspace, Bach_helper.call_price(b_test_linspace, strike, T), color='green', label='True')
plt.ylim(-3, 47)
plt.legend(loc=2, frameon=False, markerfirst=False, fontsize=7, markerscale=2)
est.set_alpha(0.4)
plt.xlabel('Spot')
plt.ylabel('Price')

plt.subplot(1, 2, 2)
est, = plt.plot(b_test, np.sum(yhat_grad, axis=1), color='red', marker="x", ls='none', ms=3, mew=0.5, label='Estimated')
plt.plot(b_test_linspace, Bach_helper.call_delta(b_test_linspace, strike, T), color='green', label='True')
plt.ylim(-0.15, 1.15)
plt.legend(loc=2, frameon=False, markerfirst=False, fontsize=7, markerscale=1.2)
est.set_alpha(0.4)
plt.xlabel('Spot')
plt.ylabel('Delta')

# plt.savefig('Graphs/ANN_price_delta.png', dpi=1000, bbox_inches='tight')
plt.show()
'''

'''
# Plot of bachelier pricing function and delta using differential ANN
plt.rc('font', size=8)
plt.figure(figsize=(9, 2))
plt.subplot(1, 2, 1)
plt.scatter(b, y, alpha=0.1, s=20)
est, = plt.plot(b_test, yhat_diff, color='red', marker="x", ls='none', ms=3, mew=0.5, label='Estimated')
plt.plot(b_test_linspace, Bach_helper.call_price(b_test_linspace, strike, T), color='green', label='True')
plt.ylim(-3, 47)
plt.legend(loc=2, frameon=False, markerfirst=False, fontsize=7, markerscale=1.2)
est.set_alpha(0.4)
plt.xlabel('Spot')
plt.ylabel('Price')

plt.subplot(1, 2, 2)
plt.scatter(b, np.sum(D, axis=1), alpha=0.1, s=20)
est, = plt.plot(b_test, np.sum(yhat_diff_grad, axis=1), color='red', marker="x", ls='none', ms=3, mew=0.5, label='Estimated')
plt.plot(b_test_linspace, Bach_helper.call_delta(b_test_linspace, strike, T), color='green', label='True')
plt.ylim(-0.15, 1.15)
plt.legend(loc=2, frameon=False, markerfirst=False, fontsize=7, markerscale=1.2)
est.set_alpha(0.4)
plt.xlabel('Spot')
plt.ylabel('Delta')

# plt.savefig('Graphs/diffANN_price_delta.png', dpi=1000, bbox_inches='tight')
plt.show()
'''

########################################################################################
###################################### SWAPTIONS #######################################
########################################################################################

from options import Swaption, Bermudan_Swaption
from models import Vasicek_eulerScheme, Vasicek_helper

def term_structure(sigma, kappa, theta):
    def p(t, T, r):
        B = (1 - tf.exp(-kappa * (T - t))) / kappa
        A = (B - T + t) * (theta - 0.5 * sigma ** 2 / kappa ** 2) - (sigma * B) ** 2 / (4 * kappa)
        return tf.exp(A - B * r)
    return p

# Set parameters
n = 50000  # Number of samples
strike = 0.1  # Strike of basket
T = np.array([0.1, 1])  # Time-to-maturity of basket call
spot_rng = np.array([0.1])
vol = 0.05  # Volatility in the model
kappa = 0.4
theta = 0.0
settlement_dates = tf.range(2, 11, dtype=tf.float32)
settlement_dates_np = np.arange(2, 11)

model = Vasicek_eulerScheme(kappa, theta, vol**2)
helper = Vasicek_helper(vol, kappa, theta)
ts = term_structure(vol, kappa, theta)

#option = Swaption(strike, T, settlement_dates, ts)
option = Bermudan_Swaption(strike, T, settlement_dates, ts)

# Create data
x, y, z = simulate_data(n, spot_rng, option, model) # seed 2
#x_linspace = np.linspace(spot_rng[0], spot_rng[1], 50).reshape(-1, 1)

print(np.mean(y))
print(helper.swaption_price(1, strike, settlement_dates_np, spot_rng))
print(helper.swaption_price(0.2, strike, settlement_dates_np, spot_rng))


'''

epsilon = 10**(-6)
est_delta = (helper.swaption_price(T, strike, settlement_dates_np, x_linspace + epsilon) -
             helper.swaption_price(T, strike, settlement_dates_np, x_linspace - epsilon)) \
            / (2 * epsilon)

# Fit model
NN = create_and_fit_diffANN(x, y, z)

# Predict and find gradients
yhat, yhat_grad = NN.predict(x_linspace)

plt.rc('font', size=8)
plt.figure(figsize=(18, 4))
plt.subplot(1, 2, 1)
plt.scatter(x, y, alpha=0.1, s=20)
plt.plot(x_linspace, yhat, color='red', label='Estimated')
plt.plot(x_linspace, helper.swaption_price(T, strike, settlement_dates_np, x_linspace), color='green', ls='dashed', label='True')
plt.legend(loc=1, frameon=False, markerfirst=False, fontsize=8)
plt.ylabel('Price')
plt.xlabel('Spot rate')

plt.subplot(1, 2, 2)
plt.scatter(x, z, alpha=0.1, s=20)
plt.plot(x_linspace, yhat_grad, color='red', label='Estimated')
plt.plot(x_linspace, est_delta, color='green', ls='dashed', label='True')
plt.legend(loc=4, frameon=False, markerfirst=False, fontsize=8)
plt.xlabel('Spot rate')
plt.ylabel('Delta')

# plt.savefig('Graphs/diffANN_price_delta_swaption.png', dpi=1000, bbox_inches='tight')
plt.show()
'''

########################################################################################
################################### DIGITAL OPTION #####################################
########################################################################################
'''
from ANN import NeuralNetwork
strike = 100
T = 0.1
spot_rng = np.array([50, 150])
rf_rate = 0.02
vol = 0.2

option = Digital(strike, T, epsilon=5)
call = Call(strike, T)
model = BlackScholes_eulerScheme(rf_rate, vol**2)
helper = BlackScholes_helper(rf_rate, vol)

n = 1000
x_call, y_call, z_call = simulate_data(n, spot_rng, call, model)
x, y, z = simulate_data(n, spot_rng, option, model)
x_linspace = np.linspace(min(spot_rng), max(spot_rng), 100)

#nn = create_and_fit_diffANN(x, y, z)
nn_call = NeuralNetwork()
nn_call.compile_fit(x_call, [y_call, z_call])

nn_non_transfer = NeuralNetwork()
nn_non_transfer.compile_fit(x, [y, z], batch_size=50)
yhat_non_transfer, yhat_grad_non_transfer = nn_non_transfer.predict(x_linspace)

nn = NeuralNetwork()
nn.compile(x, [y, z])
nn.model.build((1, 1))
nn.transfer_weights(nn_call)
nn.fit(x, [y, z], batch_size=50)

yhat, yhat_grad = nn.predict(x_linspace)

plt.subplot(1, 2, 1)
plt.scatter(x, y, s=20, alpha=0.4)
plt.plot(x_linspace, yhat, color='red')
plt.plot(x_linspace, yhat_non_transfer, color='black')
#plt.plot(x_linspace, yhat_frozen, color='black')
plt.plot(x_linspace, helper.digital_price(x_linspace, strike, T), color='green')

plt.subplot(1, 2, 2)
plt.scatter(x, z, s=20, alpha=0.4)
plt.plot(x_linspace, yhat_grad, color='red')
plt.plot(x_linspace, yhat_grad_non_transfer, color='black')
plt.plot(x_linspace, helper.digital_delta(x_linspace, strike, T), color='green')

plt.show()
'''

# fuzzy logic digital option
'''
strike = 100
epsilon = 10
x = np.linspace(50, 150, 1001)
y = np.where(x >= strike, 1, 0)
y0 = np.where(x >= strike, np.nan, 0)
y1 = np.where(x >= strike, 1, np.nan)

a = 1 / (2 * epsilon)
b = (epsilon - strike) / (2 * epsilon)
y_smoothed = np.minimum(1, np.maximum(0, a * x + b))

fig = plt.figure(figsize=(8, 5))
ax = fig.subplots(1)
plt.plot(x, y_smoothed, color='red', label='Smoothed payoff')
#plt.plot(x, y, color='black', ls=(0, (5, 5)))
plt.plot(x, y0, color='black', ls='dashed', label='Original payoff')
plt.plot(x, y1, color='black', ls='dashed')
plt.legend(loc=2, markerfirst=False, frameon=False, fontsize=9)
plt.annotate(r'K-$\epsilon$', (100-16, 0.02))
plt.annotate(r'K+$\epsilon$', (100+9, 1.01))
plt.xlabel("Spot at maturity")
plt.ylabel("Payoff")
ax.set_xticklabels([])
ax.set_xticks([])

# plt.savefig('Graphs/smoothed_digital_payoff.png', dpi=1000, bbox_inches='tight')
plt.show()
'''

'''
# Fuzzy logic digital option delta estimation
strike = 100
T = 1
spot_rng = np.array([100])
rf_rate = 0.0
vol = 0.2

option = Digital(strike, T, epsilon=2)
model = BlackScholes_eulerScheme(rf_rate, vol**2)
helper = BlackScholes_helper(rf_rate, vol)
linear = Linear(T)

true = helper.digital_delta(spot_rng[0], strike, T)

reps = 30
ns = np.array([[2**i * reps, 2**(i + 0.6) * reps] for i in range(9, 21)], dtype=int).flatten()
dn = np.array(ns / reps, dtype=int)

tmp_est = np.zeros(reps)
est = np.zeros(len(ns))
est_sd = np.zeros(len(ns))
for i in range(len(ns)):
    x, y, z = simulate_data(ns[i], spot_rng, option, model)
    for j in range(reps):
        tmp_est[j] = (np.mean(z[(j * dn[i]):((j + 1) * dn[i])]) - true) / true
    est[i] = tmp_est[0]
    est_sd[i] = np.std(tmp_est)

print(ns / reps)

plt.rc('font', size=8)
fig = plt.figure(figsize=(9, 2))
ax = fig.add_subplot(1, 2, 1)
#plt.subplot(1, 2, 1)
plt.plot(ns / reps, est, color='red', marker="x", ls='none', ms=3, mew=0.5, label='Estimated')
plt.plot(ns / reps, est, color='red', linewidth=0.2)
plt.axhline(y=0.0, ls='dashed', color='green', linewidth=0.5)
plt.legend(frameon=False, markerfirst=False, fontsize=8, markerscale=2)
plt.xscale('log', base=2)
plt.xlabel('$n$')
plt.ylabel('Deviation in %')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0, xmax=1.0))

#plt.subplot(1, 2, 2)
ax = fig.add_subplot(1, 2, 2)
plt.plot(ns / reps, est_sd, color='red', marker="x", ls='none', ms=3, mew=0.5)
plt.plot(ns / reps, est_sd, color='red', linewidth=0.2)
plt.xscale('log', base=2)
plt.xlabel('$n$')
plt.ylabel('Standard Deviation')
plt.axhline(y=0.0, ls='dashed', color='green', label='True', linewidth=0.5)

plt.subplots_adjust(wspace=0.25)
# plt.savefig('Graphs/smoothed_digital_estimationerror.png', dpi=1000, bbox_inches='tight')
plt.show()
'''

'''
# Fuzzy logic barrier 
helper = BlackScholes_helper(0.02, 0.2)

path = helper.simulate_price_path(1, 100, 1, seed=4).squeeze()
x = np.linspace(0, 1, 252)
x_extended = np.linspace(-0.2, 1.2, 250)
max_index = np.argmax(path)
barrier = path[max_index]
barrier_low = np.repeat(barrier - 2, 250)
barrier_high = np.repeat(barrier + 2, 250)

plt.figure(figsize=(8, 5))
plt.plot(x, path, color='black', label='Value of underlying')
#plt.plot(x_extended, barrier_high, color='red', ls='dashed', linewidth=1)
#plt.plot(x_extended, barrier_low, color='red', ls='dashed', linewidth=1)
fill = plt.fill_between(x_extended, barrier_low, barrier_high, color='red', alpha=0.3, label='Barrier')
plt.scatter(x[max_index], path[max_index], facecolors='none', edgecolors='red', s=170)
plt.xlim((-0.05, 1.05))
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend(loc=3, markerfirst=False, frameon=False, fontsize=9)
fill.set_alpha(0.1)
# plt.savefig('Graphs/smoothed_barrier.png', dpi=1000, bbox_inches='tight')
plt.show()
'''

########################################################################################
#################################### HEDGE ERROR #######################################
########################################################################################
'''
from models import simulate_data, BlackScholes_eulerScheme, BlackScholes_helper
from options import Call
import matplotlib.pyplot as plt
import time

## Black-Scholes call option MSE investigation
def preprocess_data(sim_sizes, rf_rate=0, vol=0.15, strike=100, T=1, nhedge=26):
    dt = T / nhedge
    spot_rng = np.array([15, 450])
    model = BlackScholes_eulerScheme(rf_rate, vol**2, antithetic=True)
    x_datasets = []
    y_datasets = []
    z_datasets = []

    for j in range(nhedge):
        option = Call(strike, T - j * dt)
        seed = (int(time.time_ns() / 1000 % 1000))
        x, y, z = simulate_data(sim_sizes[0], spot_rng, option, model, seed=seed)
        x_datasets.append(x)
        y_datasets.append(y)
        z_datasets.append(z)
        for i in range(1, len(sim_sizes)):
            seed = (int(time.time_ns() / 1000 % 1000))
            x_tmp, y_tmp, z_tmp = simulate_data(sim_sizes[i], spot_rng, option, model, seed=seed)
            x = np.row_stack((x_datasets[i-1], x_tmp))
            y = np.row_stack((y_datasets[i-1], y_tmp))
            z = np.row_stack((z_datasets[i-1], z_tmp))
            x_datasets.append(x)
            y_datasets.append(y)
            z_datasets.append(z)

    nns = [create_and_fit_diffANN(x_datasets[i], y_datasets[i], z_datasets[i], batch_size=100) for i in range(len(x_datasets))]
    #nns = [create_and_fit_ANN(x_datasets[i], y_datasets[i]) for i in range(len(x_datasets))]

    nns = np.array(nns, dtype=object).reshape(nhedge, len(sim_sizes))
    #nns = np.array(nns, dtype=object).reshape(nhedge, len(sim_sizes))
    return nns


def get_hedgeerror(nns, n=1000, spot=100., T=1, nhedge=26, rf_rate=0, vol=0.15, mu=0.05, strike=100):
    St = np.repeat(spot, n).reshape(-1, 1)
    dt = T / nhedge
    helper = BlackScholes_helper(rf_rate, vol)
    initialoutlay = helper.call_price(spot, strike, T)
    Vpf = np.repeat(initialoutlay, n).reshape(-1, 1)

    #a = helper.call_delta(St, strike, T)
    _, a = nns[0].predict(St)
    
    #St = tf.convert_to_tensor(St)
    #with tf.GradientTape() as tape:
    #    tape.watch(St)
    #    p = nns[0](St)
    #a = tape.gradient(p, St)
    
    b = Vpf - a * St

    for i in range(1, nhedge):
        St *= np.exp((mu - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * np.random.standard_normal([n, 1]))
        Vpf = a * St + b * np.exp(dt * rf_rate)

        #a = helper.call_delta(St, strike, T - i * dt)
        _, a = nns[i].predict(St)
        
        #with tf.GradientTape() as tape:
        #    tape.watch(St)
        #    p = nns[i](St)
        #a = tape.gradient(p, St)
        
        b = Vpf - a * St

    St *= np.exp((mu - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * np.random.standard_normal([n, 1]))

    Vpf = a * St + b * np.exp(dt * rf_rate)
    optionpayoff = np.maximum(St - strike, 0)
    hedgerror = Vpf - optionpayoff

    return np.std(hedgerror)


tf.config.set_visible_devices([], 'GPU')
t0 = time.time()

sim_sizes = [2**i for i in range(7, 15)]
#nns = np.zeros((26, len(sim_sizes)))

hedge_errors = np.zeros(len(sim_sizes))
for i in range(len(hedge_errors)):
    nns = preprocess_data([sim_sizes[i]])
    hedge_errors[i] = get_hedgeerror(nns[:, 0])

t1 = time.time()

print(hedge_errors)
print('Time elapsed:', t1 - t0)

plt.scatter(sim_sizes, hedge_errors, marker='x')
plt.plot(sim_sizes, hedge_errors, linewidth=0.6)
plt.axhline(y=1., color='red', ls='dotted')
plt.xscale('log', base=2)
plt.show()
'''

'''
saved_ana = np.array([1.])
saved_diff = np.array([3.05729836, 2.2709548, 1.82926648, 1.56151337, 1.48145072, 1.14395264, 1.09038785, 1.01544476])
saved = np.array([8.52156383, 7.62143133, 5.47523279, 4.7780582, 3.52229623, 3.60037621, 2.55811498, 1.87831712])
ns = [2**i for i in range(7, 15)]

plt.figure(figsize=(7, 4))
plt.scatter(ns, saved_diff, marker='x', color='red', label='DNN')
plt.scatter(ns, saved, marker='x', label='ANN')
plt.plot(ns, saved, linewidth=0.6)
plt.plot(ns, saved_diff, linewidth=0.6, color='red')
plt.axhline(y=saved_ana, color='green', ls='dotted', label='Theoretical')
plt.xscale('log', base=2)
plt.ylim((min(saved_diff) - 1, max(saved) + 0.2))
plt.xlabel('n')
plt.ylabel('Hedge error')
plt.legend(frameon=False, markerfirst=True)
# plt.savefig('Graphs/delta_hedging.png', dpi=1000, bbox_inches='tight')
plt.show()
'''

########################################################################################
################################### TIME REGRESSION ####################################
########################################################################################
'''
from sklearn.linear_model import LinearRegression

option = Call(100, 1)
model = BlackScholes_eulerScheme(0.02, 0.2**2)

#ns = [5000 * i for i in range(1, 21)]
ns = [2**i for i in range(7, 16)]

ts_dnn = []
for n in ns:
    x, y, z = simulate_data(n, [50, 150], option, model)
    t0 = time.time()
    nn = NeuralNetwork()
    nn.compile_fit(x, [y, z], epochs=100, early_stopping=False)
    t1 = time.time()
    ts_dnn.append(t1 - t0)

ts_ann = []
for n in ns:
    x, y, z = simulate_data(n, [50, 150], option, model)
    t0 = time.time()
    nn = NeuralNetwork()
    nn.compile_fit(x, [y], epochs=100, early_stopping=False)
    t1 = time.time()
    ts_ann.append(t1 - t0)


linreg_dnn = LinearRegression()
linreg_ann = LinearRegression()
X = np.array(ns).reshape(-1, 1)
y_dnn = np.array(ts_dnn).reshape(-1, 1)
y_ann = np.array(ts_ann).reshape(-1, 1)

linreg_dnn.fit(X, y_dnn)
linreg_ann.fit(X, y_ann)

print(linreg_dnn.intercept_, linreg_dnn.coef_)
print(linreg_ann.intercept_, linreg_ann.coef_)

def linear_function(x, a, b):
    return (a * x + b).squeeze()

x_linspace = np.linspace(min(ns), max(ns), 100)
plt.figure(figsize=(7, 4))
plt.scatter(ns, ts_dnn, marker='x', color='red', label='DNN')
plt.plot(x_linspace, linear_function(x_linspace, linreg_dnn.coef_, linreg_dnn.intercept_),
         color='red', linewidth=0.6)

plt.scatter(ns, ts_ann, marker='x', label='ANN')
plt.plot(x_linspace, linear_function(x_linspace, linreg_ann.coef_, linreg_ann.intercept_),
        linewidth=0.6)

plt.legend(loc=2, markerfirst=False, frameon=False)
plt.xlabel('n')
plt.ylabel('Time in seconds')
plt.xscale('log', base=2)
# plt.savefig('Graphs/training_time.png', dpi=1000, bbox_inches='tight')
plt.show()
'''

########################################################################################
###################################### ENSEMBLES #######################################
########################################################################################

'''
from ANN import Ensemble
from models import BlackScholes_helper, Bachelier_helper
# Set parameters
n = 1000  # Number of samples
n_test = 500  # Number of samples for testing fit
strike = 100  # Strike of basket
rf_rate = 0.0  # Risk-free rate
T = 0.3  # Time-to-maturity of basket call
spot_rng = [50, 150]
vol = 0.2  # Volatility in the model

vol_bach = 50  # Volatility in the model
m = 5
w = np.array([1/m for _ in range(m)])  # Weight of individual stock in basket
cov = np.identity(m) * (vol_bach ** 2)  # Covariance matrix governing stocks in basket
basket_vol = np.sqrt(np.dot(np.transpose(w), np.dot(cov, w)))

BS_helper = BlackScholes_helper(rf_rate, vol)
Bach_helper = Bachelier_helper(rf_rate, basket_vol)

#option = Call(strike, T)
option = Call_basket(strike, T, w)
#model = BlackScholes_eulerScheme(rf_rate, vol**2)
model = Bachelier_eulerScheme(rf_rate, cov)

# Create data
x, y, D = simulate_data([n, m], spot_rng, option, model)
b = np.reshape(x @ w.T, (-1, 1))
x_test, y_test, D_test = simulate_data([n_test, m], spot_rng, option, model)
b_test = np.reshape(x_test @ w.T, (-1, 1))

n = 5
bag_of_models = Ensemble(n)
bag_of_models.fit(x, y, D, full_data=0.2)

yhat_bag = bag_of_models.predict(x_test)
yhat_bag_grad = bag_of_models.delta(x_test)

MSE = np.zeros(n)
for i in range(n):
    MSE[i] = ((bag_of_models.predict_from_single(x_test, i) - Bach_helper.call_price(b_test, strike, T)) ** 2).mean()
    print("MSE from model {0}: {1}".format(i, MSE[i]))

MSE_bag = ((yhat_bag - Bach_helper.call_price(b_test, strike, T)) ** 2).mean()
print("Bagged MSE", MSE_bag)
'''
'''
# Plot of bachelier pricing function and the resulting delta using regular ANN
plt.rc('font', size=8)
plt.figure(figsize=(12, 5))
plt.subplot(2, 2, 1)
plt.scatter(b, y, alpha=0.1, s=20)
plt.plot(b_test, yhat, color='red', alpha=0.4, marker="x", ls='none', ms=3, mew=0.5, label='Estimated')
plt.plot(b_test, Bach_helper.call_price(b_test, strike, T), color='green', alpha=0.4, marker="x", ls='none', ms=3, mew=0.5, label='Analytic')
plt.ylim(-3, 47)
plt.legend(loc=2, frameon=False, markerfirst=False, fontsize=7, markerscale=2)

plt.subplot(2, 2, 2)
plt.plot(b_test, np.sum(yhat_grad, axis=1), color='red', alpha=0.4, marker="x", ls='none', ms=3, mew=0.5, label='Estimated')
plt.plot(b_test, Bach_helper.call_delta(b_test, strike, T), color='green', alpha=0.4, marker="x", ls='none', ms=3, mew=0.5, label='Analytic')
plt.ylim(-0.15, 1.15)
plt.legend(loc=2, frameon=False, markerfirst=False, fontsize=7, markerscale=2)

plt.subplot(2, 2, 3)
plt.scatter(b, y, alpha=0.1, s=20)
plt.plot(b_test, yhat_bag, color='red', alpha=0.4, marker="x", ls='none', ms=3, mew=0.5, label='Estimated')
plt.plot(b_test, Bach_helper.call_price(b_test, strike, T), color='green', alpha=0.4, marker="x", ls='none', ms=3, mew=0.5, label='Analytic')
plt.ylim(-3, 47)
plt.legend(loc=2, frameon=False, markerfirst=False, fontsize=7, markerscale=2)

plt.subplot(2, 2, 4)
plt.plot(b_test, np.sum(yhat_bag_grad, axis=1), color='red', alpha=0.4, marker="x", ls='none', ms=3, mew=0.5, label='Estimated')
plt.plot(b_test, Bach_helper.call_delta(b_test, strike, T), color='green', alpha=0.4, marker="x", ls='none', ms=3, mew=0.5, label='Analytic')
plt.ylim(-0.15, 1.15)
plt.legend(loc=2, frameon=False, markerfirst=False, fontsize=7, markerscale=2)

plt.show()
'''

########################################################################################
############################# PRINCIPAL COMPONENT ANALYSIS #############################
########################################################################################

# PCA on simulated data to show inappropriate with spreads
'''
# Initialize data
n = np.array([200, 2])
spot_rng = np.array([95, 105])
rf_rate = 0.02
vol = 0.2
cov = np.array([[vol**2, 0.99*vol**2], [0.99*vol**2, vol**2]])
T = 0.1

model = BlackScholes_eulerScheme(rf_rate, cov)
'''

'''
option = Linear(T)
X, Y, Z = simulate_data(n, spot_rng, option, model, seed=9)  # seed 9

# Plot ...
mean = Y.mean(axis=0)
minimum = Y.min() - 1
maximum = Y.max() + 1

pca2 = PCA(2, center=False)
pca2.fit(Y)
X2D = pca2.transform(Y)
basis2 = np.row_stack((pca2.basis.T, [0, 0]))
basis2 = basis2[[0, 2, 1]]

pca1 = PCA(1, center=False)
X1D = pca1.fit_transform(Y)
basis1 = np.row_stack((pca1.basis.T, [0, 0]))
slope = basis1[0, 1] / basis1[0, 0]
slope2 = - 1/slope

X1D_inv = pca1.inverse_transform(X1D)

plt.figure(figsize=(6, 6))
plt.scatter(Y[:, 0], Y[:, 1], alpha=0.5)
plt.axline(mean, slope=slope, color='black', linewidth=0.5)
plt.axline(mean, slope=slope2, color='black', linewidth=0.5, ls=(0, (5, 10)))
for i in range(len(X1D)):
    p = np.row_stack((X1D_inv[i], Y[i]))
    plt.plot(p[:, 0], p[:, 1], linewidth=0.1, color='red')
plt.plot(X1D_inv[:, 0], X1D_inv[:, 1], marker='+', color='red', alpha=0.4, mew=0.5, ls='none')
plt.xlim((minimum, maximum))
plt.ylim((minimum, maximum))
plt.xlabel('X1')
plt.ylabel('X2')
# plt.savefig('Graphs/spread_sim.png', dpi=1000, bbox_inches='tight')
plt.show()
'''


# Differential PCA
'''
option = Spread(T)
_, _, Z = simulate_data(n, spot_rng, option, model, seed=9)  # seed 9
linear_payoff = Linear(T)
_, Y, _ = simulate_data(n, spot_rng, linear_payoff, model, seed=9)  # seed 9

# Plot pathwise differentials
mean = Z.mean(axis=0)
minimum = Z.min() - 1
maximum = Z.max() + 1

pca2 = PCA(2, center=False)
pca2.fit(Z)
X2D = pca2.transform(Z)
basis2 = np.row_stack((pca2.basis.T, [0, 0]))
basis2 = basis2[[0, 2, 1]]

pca1 = PCA(1, center=False)
X1D = pca1.fit_transform(Z)
basis1 = np.row_stack((pca1.basis.T, [0, 0]))
slope = basis1[0, 1] / basis1[0, 0]
slope2 = - 1/slope
print("Suggested basis: \n{}".format(pca1.basis))

Y_minimum = Y.min() - 1
Y_maximum = Y.max() + 1
Y_mean = Y.mean(axis=0)
X1D_inv = pca1.inverse_transform(X1D)
Y1D = pca1.transform(Y)
Y1D_inv = pca1.inverse_transform(Y1D) + Y_mean
'''


'''
plt.figure(figsize=(6, 6))
plt.scatter(Z[:, 0], Z[:, 1], alpha=0.5)
plt.axline(mean, slope=slope, color='black', linewidth=0.5)
plt.axline(mean, slope=slope2, color='black', linewidth=0.5, ls=(0, (5, 10)))
for i in range(len(X1D)):
    p = np.row_stack((X1D_inv[i], Z[i]))
    plt.plot(p[:, 0], p[:, 1], linewidth=0.1, color='red')
plt.plot(X1D_inv[:, 0], X1D_inv[:, 1], marker='+', color='red', alpha=0.4, mew=0.5, ls='none')
plt.xlim((minimum, maximum))
plt.ylim((minimum, maximum))
plt.xlabel('dX1')
plt.ylabel('dX2')
# plt.savefig('Graphs/spread_diff_sim.png', dpi=1000, bbox_inches='tight')
plt.show()
'''

'''
plt.figure(figsize=(6, 6))
plt.scatter(Y[:, 0], Y[:, 1], alpha=0.5)
plt.axline(Y_mean, slope=slope, color='black', linewidth=0.5)
plt.axline(Y_mean, slope=slope2, color='black', linewidth=0.5, ls=(0, (5, 10)))
for i in range(len(Y1D_inv)):
    p = np.row_stack((Y1D_inv[i], Y[i]))
    plt.plot(p[:, 0], p[:, 1], linewidth=0.1, color='red')
plt.plot(Y1D_inv[:, 0], Y1D_inv[:, 1], marker='+', color='red', alpha=0.4, mew=0.5, ls='none')
plt.xlim((Y_minimum, Y_maximum))
plt.ylim((Y_minimum, Y_maximum))
plt.xlabel('X1')
plt.ylabel('X2')
# plt.savefig('Graphs/spread_diff_Xproj_sim.png', dpi=1000, bbox_inches='tight')
plt.show()
'''

'''
# Differential PCA in Bachelier model
n = np.array([200, 2])
spot_rng = np.array([95, 105])
rf_rate = 0.0
vol = 40
cov = np.array([[vol**2, 0.99*vol**2], [0.99*vol**2, vol**2]])
T = 0.1
w = np.array([0.7, 0.3])
strike = 100

model = Bachelier_eulerScheme(rf_rate, cov)
option = Linear(T)
call_basket = Call_basket(strike, T, w)
_, Y, _ = simulate_data(n, spot_rng, option, model, seed=9)
_, _, Z = simulate_data(n, spot_rng, call_basket, model, seed=9)

pca1d = PCA(1, center=False)
pca1d.fit(Z)
Y_proj = pca1d.transform(Y)
Y_reconstructed = pca1d.inverse_transform(Y_proj)
#Y_reconstructed = pca1d.reconstruct(Y, center=False)
b = pca1d.basis
print(b)
print(w / np.linalg.norm(w))
slope = b[1] / b[0]

point1 = Y_reconstructed[5]
point2 = np.copy(point1)
point2[0] += 21
point3 = np.copy(point2)
point3[1] += 9

points = np.row_stack((point1, point2, point3))
'''
'''
plt.figure(figsize=(7, 5))
datapoints = plt.scatter(Y[:, 0], Y[:, 1], label='Data')
crosses, = plt.plot(Y_reconstructed[:, 0], Y_reconstructed[:, 1], marker='+', color='red', mew=0.5, ls='none', label='Projection')
for i in range(len(Y_reconstructed)):
    p = np.row_stack((Y_reconstructed[i], Y[i]))
    plt.plot(p[:, 0], p[:, 1], linewidth=0.1, color='red', alpha=0.5, ls='dashed')
plt.axline([0, 0], slope=slope, color='green', linewidth=1, ls='solid', label='Theoretical')
plt.xlim((30, 180))
plt.ylim((30, 135))
plt.legend(loc=2, frameon=False, markerfirst=False)
crosses.set_alpha(0.5)
datapoints.set_alpha(0.4)
plt.plot(points[:, 0], points[:, 1], ls=(0, (1, 1)), linewidth=1.2)
plt.annotate('0.7', (point1-[-9, 4]))
plt.annotate('0.3', (point2+[2, 4]))
plt.xlabel('$S_1(t)$')
plt.ylabel('$S_2(t)$')
# plt.savefig('Graphs/Bachelier_2dim_reduction.png', dpi=1000, bbox_inches='tight')
plt.show()
'''

'''
# Black-Scholes call on basket with DPCA
# Set parameters
n = 1000  # Number of samples
n_test = 1000  # Number of samples for testing fit
strike = 100  # Strike of basket
rf_rate = 0.02  # Risk-free rate
T = 0.3  # Time-to-maturity of basket call
spot_rng = [50, 150]


vol = 0.2  # Volatility in the model
var = vol**2
m = 6
w = np.array([1/m for _ in range(m)])  # Weight of individual stock in basket
#w = np.array([0.1, 0.15, 0.3, 0.4, 0.05])
#cov = np.identity(m) * (vol ** 2)  # Covariance matrix governing stocks in basket
np.random.seed(1)
cov = np.random.uniform(-1, 1, (m, m))
cov = np.round((cov.T @ cov) * 0.02, 3)
print(cov)


#basket_vol = np.sqrt(np.dot(np.transpose(w), np.dot(cov, w)))

# Simulate datasets for training and testing
option = Call_basket(strike, T, w)
#Bach_helper = Bachelier_helper(rf_rate, basket_vol)

model = BlackScholes_eulerScheme(rf_rate, cov)

x, y, z = simulate_data([n, m], spot_rng, option, model, seed=2)
b = np.reshape(x @ w.T, (-1, 1))

x_test, y_test, z_test = simulate_data([n_test, m], spot_rng, option, model)
b_test = np.reshape(x_test @ w.T, (-1, 1))
order = np.argsort(b_test, axis=0).flatten()
x_test = x_test[order]
b_test = b_test[order]
b_test_linspace = np.linspace(min(b_test), max(b_test), 100)

pca = PCA(1, center=False)
z1d, x1d = pca.fit_transform(z, x)
x1d_test = pca.transform(x_test)
print(pca.eigen_values)
print(pca.explained_variance)


nn_pca = create_and_fit_diffANN(x1d, y, z1d)
yhat_pca, yhat_pca_grad = nn_pca.predict(x1d_test)
yhat_pca_grad = pca.inverse_transform(yhat_pca_grad)

nn = create_and_fit_diffANN(x, y, z)
yhat, yhat_grad = nn.predict(x_test)
'''

'''
# Plot of bachelier pricing function and delta using differential ANN
plt.rc('font', size=8)
plt.figure(figsize=(9, 2))
plt.subplot(1, 2, 1)
plt.scatter(b, y, alpha=0.1, s=20)
est, = plt.plot(b_test, yhat, color='red', marker="x", ls='none', ms=3, mew=0.5, label='Estimated w/o DPCA')
plt.plot(b_test, yhat_pca, color='green', label='Estimated w/ DPCA(1)')
plt.ylim(-3, 47)
plt.legend(loc=2, frameon=False, markerfirst=False, fontsize=7, markerscale=1.2)
est.set_alpha(0.4)
plt.ylabel('Price')
plt.xlabel('Spot')

plt.subplot(1, 2, 2)
plt.scatter(b, np.sum(z, axis=1), alpha=0.1, s=20)
est, = plt.plot(b_test, np.sum(yhat_grad, axis=1), color='red', marker="x", ls='none', ms=3, mew=0.5, label='Estimated w/o DPCA')
plt.plot(b_test, np.sum(yhat_pca_grad, axis=1), color='green', label='Estimated w/ DPCA(1)')
plt.ylim(-0.15, 1.15)
plt.legend(loc=2, frameon=False, markerfirst=False, fontsize=7, markerscale=1.2)
est.set_alpha(0.4)
plt.ylabel('Delta')
plt.xlabel('Spot')

# plt.savefig('Graphs/BS_call_basket_DPCA.png', dpi=1000, bbox_inches='tight')
plt.show()
'''
'''
# Plot of bachelier pricing function and delta using differential ANN
plt.rc('font', size=8)
plt.figure(figsize=(9, 2))
plt.subplot(1, 2, 1)
plt.scatter(b, y, alpha=0.1, s=20)
plt.plot(b_test, yhat_diff, color='red', label='Estimated')
plt.plot(b_test_linspace, Bach_helper.call_price(b_test_linspace, strike, T), color='green', label='True')
plt.ylim(-3, 47)
plt.legend(loc=2, frameon=False, markerfirst=False, fontsize=7, markerscale=1.2)
plt.xlabel('Spot')
plt.ylabel('Price')

plt.subplot(1, 2, 2)
plt.scatter(b, np.sum(z, axis=1), alpha=0.1, s=20)
plt.plot(b_test, np.sum(yhat_diff_grad, axis=1), color='red', label='Estimated')
plt.plot(b_test_linspace, Bach_helper.call_delta(b_test_linspace, strike, T), color='green', label='True')
plt.ylim(-0.15, 1.15)
plt.legend(loc=2, frameon=False, markerfirst=False, fontsize=7)
plt.xlabel('Spot')
plt.ylabel('Delta')

# plt.savefig('Graphs/diffANN_price_delta_DPCA.png', dpi=1000, bbox_inches='tight')
plt.show()
'''




########################################################################################
################################# CONVERGENCE ORDER ####################################
########################################################################################
'''
def mse(yhat, y):
    return np.mean(((yhat - y) / y)**2)

def rmse(yhat, y):
    return np.sqrt(np.mean(((yhat - y) / y)**2))

# Set parameters
#n = 1000  # Number of samples
n_test = 1000  # Number of samples for testing fit
strike = 100  # Strike of basket
rf_rate = 0.0  # Risk-free rate
T = 0.3  # Time-to-maturity of basket call
spot_rng = [50, 150]
vol_bs = 0.2  # Volatility in the model

# Simulate datasets for training and testing
option = Call(strike, T)
BS_helper = BlackScholes_helper(rf_rate, vol_bs)
model = BlackScholes_eulerScheme(rf_rate, vol_bs**2)

#x_test, y_test, z_test = simulate_data(n_test, spot_rng, option, model)
x_test = np.linspace(spot_rng[0], spot_rng[1], 1000).reshape(-1, 1)

m = 8
s = 3
ns = np.array([[2**(7+i), 2**(7.4+i), 2**(7.7+i)] for i in range(0, m)], dtype=np.int).flatten()
rmses = []
rmses_diff = []
true_y = BS_helper.call_price(x_test, strike, T)

for n in ns:
    yhat = 0
    yhat_diff = 0
    for i in range(s):
        x, y, z = simulate_data(n, spot_rng, option, model, seed=i)

        nn = create_and_fit_ANN(x, y)
        nn_diff = create_and_fit_diffANN(x, y, z)

        yhat += nn.predict(x_test)
        yhat_diff += nn_diff.predict(x_test)[0]

    yhat /= s
    yhat_diff /= s
    rmses.append(rmse(yhat, true_y))
    rmses_diff.append(rmse(yhat_diff, true_y))

log_rmses = np.log(rmses)
log_rmses_diff = np.log(rmses_diff)
log_ns = np.log(ns)
#print(ns)
#print(np.round(rmses, 5))

reg = sklearn.linear_model.LinearRegression()
reg.fit(log_ns.reshape(-1, 1), log_rmses)
reg_diff = sklearn.linear_model.LinearRegression()
reg_diff.fit(log_ns.reshape(-1, 1), log_rmses_diff)

intercept, slope = reg.intercept_, reg.coef_
intercept_diff, slope_diff = reg_diff.intercept_, reg_diff.coef_
print(intercept, slope)
print(intercept_diff, slope_diff)

plt.plot(log_ns, log_rmses, marker="x", linewidth=0.2, ms=5, mew=0.75)
plt.plot(log_ns, log_rmses_diff, marker="x", linewidth=0.2, ms=5, mew=0.75, color='red')
plt.axline((0, intercept), slope=slope, linewidth=0.5)
plt.axline((0, intercept_diff), slope=slope_diff, linewidth=0.5, color='red')
plt.xlim((min(log_ns)-1, max(log_ns)+1))
plt.ylim((min(log_rmses_diff)-1, max(log_rmses)+1))
# plt.savefig('Graphs/convergence_order_BS.png', dpi=1000, bbox_inches='tight')
plt.show()
'''











