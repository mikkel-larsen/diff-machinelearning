from abc import ABC

import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time

from models import BlackScholes, Bachelier, Bachelier_eulerScheme, simulate_data
from options import Call, Call_basket

class create_diffANN(keras.Model, ABC):
    def __init__(self, mean_, scale_, units=5, activation="softplus", seed=None, **kwargs):
        super().__init__(**kwargs)
        self.mean_ = mean_
        self.scale_ = scale_
        self.initializer = keras.initializers.he_normal(seed=seed)
        self.hidden1 = keras.layers.Dense(units, activation=activation, kernel_initializer=self.initializer)
        self.hidden2 = keras.layers.Dense(units, activation=activation, kernel_initializer=self.initializer)
        self.hidden3 = keras.layers.Dense(units, activation=activation, kernel_initializer=self.initializer)
        self.hidden4 = keras.layers.Dense(units, activation=activation, kernel_initializer=self.initializer)
        self.reg_output = keras.layers.Dense(1)

    def call(self, inputs):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            inputs_scaled = (inputs - self.mean_) / self.scale_
            hidden1 = self.hidden1(inputs_scaled)
            hidden2 = self.hidden2(hidden1)
            hidden3 = self.hidden3(hidden2)
            hidden4 = self.hidden4(hidden3)
            reg_output = self.reg_output(hidden4)

        grad_output = tape.gradient(reg_output, inputs)
        return reg_output, grad_output

class create_ANN(keras.Model, ABC):
    def __init__(self, mean_, scale_, units=5, activation="softplus", seed=None, **kwargs):
        super().__init__(**kwargs)
        self.mean_ = mean_
        self.scale_ = scale_
        self.initializer = keras.initializers.he_normal(seed=seed)
        self.hidden1 = keras.layers.Dense(units, activation=activation, kernel_initializer=self.initializer)
        self.hidden2 = keras.layers.Dense(units, activation=activation, kernel_initializer=self.initializer)
        self.hidden3 = keras.layers.Dense(units, activation=activation, kernel_initializer=self.initializer)
        self.hidden4 = keras.layers.Dense(units, activation=activation, kernel_initializer=self.initializer)
        self.reg_output = keras.layers.Dense(1)

    def call(self, inputs):
        inputs_scaled = (inputs - self.mean_) / self.scale_
        hidden1 = self.hidden1(inputs_scaled)
        hidden2 = self.hidden2(hidden1)
        hidden3 = self.hidden3(hidden2)
        hidden4 = self.hidden4(hidden3)
        reg_output = self.reg_output(hidden4)

        return reg_output

def create_and_fit_diffANN(x, y, D, epochs=60, batch_size=250, nodes=5, activations="softplus", seed=None):
    scaler = StandardScaler()
    scaler.fit(x)
    D_perm = np.transpose(D[:, :, np.newaxis], axes=(1, 0, 2))
    lmbda = np.dot(np.transpose(y), y) / np.dot(np.transpose(D_perm[0, :, :]), D_perm[0, :, :])
    w = lmbda / (1 + lmbda).item()

    model = create_diffANN(scaler.mean_, scaler.scale_, nodes, activations, seed)

    opt = keras.optimizers.Adam(learning_rate=0.1)
    model.compile(loss='mse', optimizer=opt, loss_weights=[1 - w, w])

    early_stopping_cb = keras.callbacks.EarlyStopping(patience=7, min_delta=0.0000001, restore_best_weights=True,
                                                      monitor='loss', verbose=1)
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=4, monitor='loss', verbose=0)
    model.fit(x, [y, D], epochs=epochs, batch_size=batch_size, callbacks=[lr_scheduler, early_stopping_cb])

    return model

def create_and_fit_ANN(x, y, epochs=60, batch_size=250, nodes=5, activations="softplus", seed=None):
    scaler = StandardScaler()
    scaler.fit(x)

    model = create_ANN(scaler.mean_, scaler.scale_, nodes, activations, seed)

    opt = keras.optimizers.Adam(learning_rate=0.1)
    model.compile(loss='mse', optimizer=opt)

    early_stopping_cb = keras.callbacks.EarlyStopping(patience=7, min_delta=0.001, restore_best_weights=True,
                                                      monitor='loss', verbose=1)
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=4, monitor='loss', verbose=0)
    model.fit(x, y, epochs=epochs, batch_size=batch_size, callbacks=[lr_scheduler, early_stopping_cb])

    return model


# Set parameters
n = 4000  # Number of samples
n_test = 3000  # Number of samples for testing fit
strike = 110  # Strike of basket
rf_rate = 0.05  # Risk-free rate
T = 0.2  # Time-to-maturity of basket option
spot_rng = [20, 200]

# ---------- BLACK-SCHOLES ----------
vol_bs = 0.2  # Volatility in the model

# ---------- BACHELIER --------------
vol_bach = 50  # Volatility in the model
m = 10
w = np.array([1/m for _ in range(m)])  # Weight of individual stock in basket
# cov = np.identity(m) * (vol_bach ** 2)  # Covariance matrix governing stocks in basket
np.random.seed(7)
cov = np.random.uniform(-100, 100, (m, m))
cov = cov.T @ cov
basket_vol = np.sqrt(np.dot(np.transpose(w), np.dot(cov, w)))

# Simulate datasets for training and testing
option = Call(strike, T)
basket = Call_basket(strike, T, w)
Bach = Bachelier(rf_rate, basket_vol)
BS = BlackScholes(rf_rate, vol_bs)

model = Bachelier_eulerScheme(rf_rate, cov)

x, y, D = simulate_data([n, m], spot_rng, basket, model)
b = np.reshape(x @ w.T, (-1, 1))

x_test, y_test, D_test = simulate_data([n_test, m], spot_rng, basket, model)
b_test = np.reshape(x_test @ w.T, (-1, 1))

# Create, fit and time models
t0 = time.time()
model = create_and_fit_ANN(x, y, epochs=25)
t1 = time.time()
training_time = t1 - t0

t0 = time.time()
model_diff = create_and_fit_diffANN(x, y, D, epochs=25)
t1 = time.time()
training_time_diff = t1 - t0

# Predict and find gradients
yhat_diff, yhat_diff_grad = model_diff.predict(x_test)

x_test = tf.convert_to_tensor(x_test)
with tf.GradientTape() as tape:
    tape.watch(x_test)
    yhat = model(x_test)

yhat_grad = tape.gradient(yhat, x_test)

# index_test = np.random.randint(0, n_test, 3)
# print("Predicted regression: \n {} \n Correct: \n {}".format(yhat[index_test], BS.call_price(x_test[index_test], strike, T)))
# print("Predicted gradient: \n {} \n Correct: \n {}".format(yhat_grad[index_test], BS.call_delta(x_test[index_test], strike, T)))

scaler = StandardScaler()
scaler.fit(x)
Z_perm = np.transpose(D[:, :, np.newaxis], axes=(1, 0, 2))
lmbda = np.dot(np.transpose(y), y) / np.dot(np.transpose(Z_perm[0, :, :]), Z_perm[0, :, :]) # Lambda for regularization, weighted by variation of term
w = (lmbda / (1 + lmbda)).item()
print("Weight: {}".format(w))
print("Training time: \n {}".format(t1-t0))
print("Basket vol: {}".format(basket_vol))

# ---------- Plot ------------------------
'''
plt.subplot(2, 1, 1)
plt.scatter(x, y, alpha=0.1)
plt.plot(x_test, BS.straddle_price(x_test, strike, T), linestyle='dotted', color='black')
plt.plot(x_test, yhat, color='red')
plt.text(50, 105, "{} sec".format(np.round(training_time, 2)), horizontalalignment='left', verticalalignment='center', fontsize=8)

plt.subplot(2, 1, 2)
plt.scatter(x, D, alpha=0.1)
plt.plot(x_test, BS.straddle_delta(x_test, strike, T), linestyle='dotted', color='black')
plt.plot(x_test, yhat_grad, color='red')

plt.show()
'''

plt.subplot(2, 2, 1)
plt.scatter(b, y, alpha=0.1)
plt.plot(b_test, yhat_diff, color='red', alpha=0.4, marker="x", ls='none', ms=3, mew=0.5)
plt.plot(b_test, Bach.call_price(b_test, strike, T), color='green', alpha=0.4, marker="x", ls='none', ms=3, mew=0.5)
plt.text(60, 80, "{} sec".format(np.round(training_time_diff, 2)), horizontalalignment='left', verticalalignment='center', fontsize=8)

plt.subplot(2, 2, 2)
plt.scatter(b, D[:, 0], alpha=0.1)
plt.plot(b_test, yhat_diff_grad[:, 0], color='red', alpha=0.4, marker="x", ls='none', ms=3, mew=0.5)
plt.plot(b_test, Bach.call_delta(b_test, strike, T) / m, color='green', alpha=0.4, marker="x", ls='none', ms=3, mew=0.5)

plt.subplot(2, 2, 3)
plt.scatter(b, y, alpha=0.1)
plt.plot(b_test, yhat, color='red', alpha=0.4, marker="x", ls='none', ms=3, mew=0.5)
plt.plot(b_test, Bach.call_price(b_test, strike, T), color='green', alpha=0.4, marker="x", ls='none', ms=3, mew=0.5)
plt.text(60, 80, "{} sec".format(np.round(training_time, 2)), horizontalalignment='left', verticalalignment='center', fontsize=8)

plt.subplot(2, 2, 4)
plt.scatter(b, D[:, 0], alpha=0.1)
plt.plot(b_test, yhat_grad[:, 0], color='red', alpha=0.4, marker="x", ls='none', ms=3, mew=0.5)
plt.plot(b_test, Bach.call_delta(b_test, strike, T) / m, color='green', alpha=0.4, marker="x", ls='none', ms=3, mew=0.5)

plt.show()