import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from models import BlackScholes

# Set parameters
n = 10000  # Number of samples
n_test = 200  # Number of samples for testing fit
strike = 100  # Strike of basket

# ---------- BLACK-SCHOLES ----------
rf_rate = 0.02  # Risk-free rate (0 to easily simulate in the Bachelier model)
vol = 0.2  # Volatility in the model
T = 0.5  # Time-to-maturity of basket option

# Simulate BS dataset for training and testing
BS = BlackScholes(rf_rate, vol)
x, y, D = BS.simulate_data(n, 50, 150, strike, T)
# y = BS.price(x, strike, T)
# y_delta = BS.delta(x, strike, T)

x_test, y_test, D_test = BS.simulate_data(n_test, 50, 150, strike, T)
y_test_delta = BS.delta(x_test, strike, T)

index = np.random.randint(0, n, 4)
print("x\n", x[index])
print("y\n", y[index])
print("delta\n", D[index])


class ANN_reg_and_grad(keras.Model):
    def __init__(self, mean_, scale_, units=5, activation="softplus", **kwargs):
        super().__init__(**kwargs)
        self.mean_ = mean_
        self.scale_ = scale_
        self.hidden1 = keras.layers.Dense(units, activation=activation, kernel_initializer="he_normal")
        self.hidden2 = keras.layers.Dense(units, activation=activation, kernel_initializer="he_normal")
        self.hidden3 = keras.layers.Dense(units, activation=activation, kernel_initializer="he_normal")
        self.hidden4 = keras.layers.Dense(units, activation=activation, kernel_initializer="he_normal")
        self.hidden5 = keras.layers.Dense(units, activation=activation, kernel_initializer="he_normal")
        self.reg_output = keras.layers.Dense(1)

    def call(self, inputs):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            inputs_scaled = (inputs - self.mean_) / self.scale_
            hidden1 = self.hidden1(inputs_scaled)
            hidden2 = self.hidden2(hidden1)
            hidden3 = self.hidden3(hidden2)
            hidden4 = self.hidden4(hidden3)
            hidden5 = self.hidden4(hidden4)
            reg_output = self.reg_output(hidden5)

        grad_output = tape.gradient(reg_output, inputs)
        return reg_output, grad_output

# scaler = StandardScaler()
# x = scaler.fit_transform(x)
# x_test = scaler.transform(x_test)
# print("X scaled (mean: {0}, std: {1}): \n {2}".format(scaler.mean_.round(3), scaler.scale_.round(3), x[index]))

scaler = StandardScaler()
scaler.fit_transform(x)
model = ANN_reg_and_grad(scaler.mean_, scaler.scale_)

opt = keras.optimizers.Adam(learning_rate=0.1)
model.compile(loss='mse', optimizer=opt, loss_weights=[0.1, 0.9])

early_stopping_cb = keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor='loss', verbose=1)
lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=4, monitor='loss', verbose=0)
model.fit(x, [y, D], epochs=60, batch_size=100, callbacks=[lr_scheduler, early_stopping_cb])

yhat, yhat_grad = model.predict(x_test)
# yhat_grad /= scaler.scale_

# x = scaler.inverse_transform(x)
# x_test = scaler.inverse_transform(x_test)
# print("X scaled back: \n", x[index])

index_test = np.random.randint(0, n_test, 4)
print("Predicted regression: \n {} \n Correct: \n {}".format(yhat[index_test], BS.price(x_test[index_test], strike, T)))
print("Predicted gradient: \n {} \n Correct: \n {}".format(yhat_grad[index_test], BS.delta(x_test[index_test], strike, T)))

plt.subplot(1, 2, 1)
plt.scatter(x, y, alpha=0.1)
plt.plot(x_test, BS.price(x_test, strike, T), linestyle='dotted', color='black')
plt.plot(x_test, yhat, color='red')

plt.subplot(1, 2, 2)
plt.scatter(x, D, alpha=0.1)
plt.plot(x_test, BS.delta(x_test, strike, T), linestyle='dotted', color='black')
plt.plot(x_test, yhat_grad, color='red')

plt.show()
