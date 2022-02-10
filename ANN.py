from abc import ABC
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from sklearn.preprocessing import StandardScaler

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

def create_and_fit_diffANN(x, y, D, epochs=60, batch_size=150, nodes=5, activations="softplus", seed=None):
    scaler = StandardScaler()
    scaler.fit(x)
    D_perm = np.transpose(D[:, :, np.newaxis], axes=(1, 0, 2))
    lmbda = np.dot(np.transpose(y), y) / np.dot(np.transpose(D_perm[0, :, :]), D_perm[0, :, :])
    w = lmbda / (1 + lmbda)

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


class Ensemble:
    def __init__(self, n):
        self.n = n
        self.models = None

    def fit(self, x, y, D, full_data=True):
        if full_data is False:
            n = int(np.shape(x)[0])
            i = np.random.choice(n, size=int(n/self.n))
            print("Ikke full", int(n/self.n))
            self.models = [create_and_fit_diffANN(x[i], y[i], D[i]) for _ in range(self.n)]
        elif type(full_data) is float:
            n = int(np.shape(x)[0])
            i = np.random.choice(n, size=int(n * full_data))
            print("procent af full", int(n * full_data))
            self.models = [create_and_fit_diffANN(x[i], y[i], D[i]) for _ in range(self.n)]
        else:
            self.models = [create_and_fit_diffANN(x, y, D) for _ in range(self.n)]

        return self.models

    def predict(self, x):
        running_sum = np.array([model.predict(x)[0] for model in self.models])
        return np.mean(running_sum.reshape(self.n, -1), axis=0).reshape(-1, 1)

    def predict_from_single(self, x, i):
        return self.models[i].predict(x)[0]

    def delta(self, x):
        running_sum = np.array([model.predict(x)[1] for model in self.models])
        return np.mean(running_sum, axis=0)

