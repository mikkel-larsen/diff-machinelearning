import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from scipy.stats import norm

from simulate_data import BS
from simulate_data import transform

def true_BS(spot, rf_rate, vol, T, strike):
    d1 = (np.log(spot / strike) + (rf_rate + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return spot * norm.cdf(d1) - strike * np.exp(-rf_rate * T) * norm.cdf(d2)

def create_ANN(x, y, epochs, lmbda, rate, nodes=5):
    # define the model
    np.random.seed(1)
    init = keras.initializers.GlorotNormal(seed=1)
    model = Sequential()
    model.add(Dense(nodes, input_shape=(1,), kernel_regularizer=keras.regularizers.l2(lmbda), activation='softplus', kernel_initializer=init))
    model.add(Dense(nodes, kernel_regularizer=keras.regularizers.l2(lmbda), activation='softplus', kernel_initializer=init))
    model.add(Dense(nodes, kernel_regularizer=keras.regularizers.l2(lmbda), activation='softplus', kernel_initializer=init))
    model.add(Dense(nodes, kernel_regularizer=keras.regularizers.l2(lmbda), activation='softplus', kernel_initializer=init))
    model.add(Dense(1, kernel_regularizer=keras.regularizers.l2(lmbda), activation='linear', kernel_initializer=init))

    # compile the model
    opt = keras.optimizers.Adam(learning_rate=rate)
    model.compile(optimizer=opt, loss='mse')

    # Callback
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            global losses
            losses.append(logs.get("loss"))

    # fit the model
    model.fit(x, y, epochs=epochs, batch_size=200)

    return model

n = 30000
strike = 100
rf_rate = 0.02
vol = 0.15
T = 0.3

model = BS(rf_rate, vol, T)

x = np.array([transform(np.linspace(20, 180, n), strike)])
sim = model.BS_simulate_endpoint(n, x)
y = np.maximum(sim - strike, 0)

x = x.reshape(n, 1)
y = y.reshape(n, 1)

'''
m = 30
loss = np.zeros(m)
for i in range(m):
    mod = create_ANN(x, y, 75, 0, 0.00001 + (1 - 0.00001) / 30 * i)
    loss[i] = mod.evaluate(x, y, verbose=0)

print("Final losses: {}".format(loss))

rates = [0.00001 + (1 - 0.00001) / 30 * i for i in range(m)]

plt.plot(rates, loss)
plt.show()
'''
'''
mod = create_ANN(x, y, 50, 0, 0.003)

# make a prediction
yhat = model.predict(x)

# plot
plt.scatter(x, y, alpha=0.1)
plt.plot(x, true_BS(x, rf_rate, vol, T, strike), linestyle='dotted', color='black')
plt.plot(x, yhat, color='red')
plt.show()
'''