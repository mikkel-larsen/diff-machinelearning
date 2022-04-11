from abc import ABC
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler

tf.config.set_visible_devices([], 'GPU')

class create_diffANN(keras.Model, ABC):
    def __init__(self, n_hidden=4, n_neurons=5, activation='softplus', seed=None, **kwargs):
        super().__init__(**kwargs)
        self.initializer = keras.initializers.he_normal(seed=seed)
        self.hidden_layers = [Dense(n_neurons, activation=activation, kernel_initializer=self.initializer)
                              for _ in range(n_hidden)]
        self.reg_output = Dense(1)

    def call(self, inputs):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            hidden = inputs
            for layer in self.hidden_layers:
                hidden = layer(hidden)
            reg_output = self.reg_output(hidden)

        grad_output = tape.gradient(reg_output, inputs)
        return reg_output, grad_output


class create_ANN(keras.Model, ABC):
    def __init__(self, mean_, scale_, n_hidden=4, n_neurons=5, activation='softplus', seed=None, **kwargs):
        super().__init__(**kwargs)
        self.mean_ = mean_
        self.scale_ = scale_
        self.initializer = keras.initializers.he_normal(seed=seed)
        self.hidden_layers = [Dense(n_neurons, activation=activation, kernel_initializer=self.initializer)
                              for _ in range(n_hidden)]
        self.output_layer = Dense(1)

    def call(self, inputs):
        hidden = (inputs - self.mean_) / self.scale_
        for layer in self.hidden_layers:
            hidden = layer(hidden)
        output = self.output_layer(hidden)

        return output


class NeuralNetwork:
    def __init__(self, **kwargs):
        self.model = None

    def __call__(self, x, **kwargs):
        x = np.array(x).reshape(-1, 1)
        return self.model(x)

    def compile(self, w=None, n_hidden=4, n_neurons=6, act='softplus', lr=0.1, seed=None, **kwargs):
        self.model = create_diffANN(n_hidden, n_neurons, act, seed)
        opt = keras.optimizers.Adam(learning_rate=lr)
        if w is None:
            self.model.compile(loss='mse', optimizer=opt, **kwargs)
        else:
            self.model.compile(loss='mse', optimizer=opt, loss_weights=[1 - w, w], **kwargs)

        return self.model

    def fit(self, x, y, epochs=100, batch_size=200, early_stopping=True, **kwargs):
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=6,
                                                         monitor='loss', verbose=0)
        if early_stopping:
            early_stopping_cb = keras.callbacks.EarlyStopping(patience=20,
                                                              min_delta=0.000001,
                                                              restore_best_weights=True,
                                                              monitor='loss', verbose=1)

            self.model.fit(x, y, epochs=epochs, batch_size=batch_size,
                           callbacks=[lr_scheduler, early_stopping_cb], verbose=0)
        else:
            self.model.fit(x, y, epochs=epochs, batch_size=batch_size,
                           callbacks=[lr_scheduler], verbose=0)
        return self.model

    def compile_fit(self, x, y, epochs=100, batch_size=200, n_hidden=4,
                    n_neurons=6, act='softplus', lr=0.1, seed=None, early_stopping=True, **kwargs):
        y1, *z = y
        y1 = np.array(y1).reshape(-1, 1)
        z = np.array(z).reshape(-1, 1)
        w = None
        if z.size > 0:
            z_perm = np.transpose(z[:, :, np.newaxis], axes=(1, 0, 2))
            lmbda = np.dot(np.transpose(y1), y1) / np.dot(np.transpose(z_perm[0, :, :]), z_perm[0, :, :])
            w = lmbda / (1 + lmbda)

        self.compile(w, n_hidden, n_neurons, act, lr, seed, **kwargs)
        self.fit(x, y, epochs, batch_size, early_stopping, **kwargs)
        return self.model

    def predict(self, x):
        #x = np.array(x, dtype=float).reshape(-1, 1)
        return self.model.predict(x)

    def transfer_weights(self, model):
        self.model.set_weights(model.model.get_weights())
        return self.model


###### Quick and dirty functions

class create_diffANN2(keras.Model, ABC):
    def __init__(self, mean_, scale_, n_hidden=4, n_neurons=5, activation='softplus', seed=None, **kwargs):
        super().__init__(**kwargs)
        self.mean_ = mean_
        self.scale_ = scale_
        self.initializer = keras.initializers.he_normal(seed=seed)
        self.hidden_layers = [Dense(n_neurons, activation=activation, kernel_initializer=self.initializer)
                              for _ in range(n_hidden)]
        self.reg_output = Dense(1)

    def call(self, inputs):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            hidden = (inputs - self.mean_) / self.scale_
            for layer in self.hidden_layers:
                hidden = layer(hidden)
            reg_output = self.reg_output(hidden)

        grad_output = tape.gradient(reg_output, inputs)
        return reg_output, grad_output


class create_ANN2(keras.Model, ABC):
    def __init__(self, mean_, scale_, n_hidden=4, n_neurons=5, activation='softplus', seed=None, **kwargs):
        super().__init__(**kwargs)
        self.mean_ = mean_
        self.scale_ = scale_
        self.initializer = keras.initializers.he_normal(seed=seed)
        self.hidden_layers = [Dense(n_neurons, activation=activation, kernel_initializer=self.initializer)
                              for _ in range(n_hidden)]
        self.output_layer = Dense(1)

    def call(self, inputs):
        hidden = (inputs - self.mean_) / self.scale_
        for layer in self.hidden_layers:
            hidden = layer(hidden)
        output = self.output_layer(hidden)

        return output

def create_and_fit_diffANN(x, y, D, epochs=60, batch_size=200, nodes=6, activations="softplus", seed=None):
    scaler = StandardScaler()
    scaler.fit(x)
    D_perm = np.transpose(D[:, :, np.newaxis], axes=(1, 0, 2))
    lmbda = np.dot(np.transpose(y), y) / np.dot(np.transpose(D_perm[0, :, :]), D_perm[0, :, :])
    w = lmbda / (1 + lmbda)

    model = create_diffANN2(scaler.mean_, scaler.scale_, 4, nodes, activations, seed)

    opt = keras.optimizers.Adam(learning_rate=0.1)
    model.compile(loss='mse', optimizer=opt, loss_weights=[1 - w, w])

    early_stopping_cb = keras.callbacks.EarlyStopping(patience=15, min_delta=0.000000001, restore_best_weights=True,
                                                      monitor='loss', verbose=1)
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=4, monitor='loss', verbose=0)
    model.fit(x, [y, D], epochs=epochs, batch_size=batch_size, callbacks=[lr_scheduler, early_stopping_cb], verbose=0)

    return model


def create_and_fit_ANN(x, y, epochs=60, batch_size=200, nodes=6, activations="softplus", seed=None):
    scaler = StandardScaler()
    scaler.fit(x)

    model = create_ANN2(scaler.mean_, scaler.scale_, 4, nodes, activations, seed)

    opt = keras.optimizers.Adam(learning_rate=0.1)
    model.compile(loss='mse', optimizer=opt)

    early_stopping_cb = keras.callbacks.EarlyStopping(patience=15, min_delta=0.000000001, restore_best_weights=True,
                                                      monitor='loss', verbose=1)
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=4, monitor='loss', verbose=0)
    model.fit(x, y, epochs=epochs, batch_size=batch_size, callbacks=[lr_scheduler, early_stopping_cb], verbose=0)

    return model


class Ensemble:
    def __init__(self, n):
        self.n = n
        self.models = None

    def __getitem__(self, item):
        return self.models[item]

    def fit(self, x, y, D, full_data=False):
        if full_data is False:
            n = int(np.shape(x)[0])
            i = np.random.choice(n, size=int(n / self.n))
            self.models = [create_and_fit_diffANN(x[i], y[i], D[i]) for _ in range(self.n)]
        elif type(full_data) is float:
            n = int(np.shape(x)[0])
            i = np.random.choice(n, size=int(n * full_data))
            self.models = [create_and_fit_diffANN(x[i], y[i], D[i]) for _ in range(self.n)]
        else:
            self.models = [create_and_fit_diffANN(x, y, D) for _ in range(self.n)]

        return self.models

    def predict(self, x):
        running_sum = np.array([model.predict(x)[0] for model in self.models])
        return np.mean(running_sum.reshape(self.n, -1), axis=0).reshape(-1, 1)

    def predict_from_single(self, x, i=0):
        return self.models[int(i)].predict(x)[0]

    def delta(self, x):
        running_sum = np.array([model.predict(x)[1] for model in self.models])
        return np.mean(running_sum, axis=0)

    def delta_from_single(self, x, i=0):
        return self.models[int(i)].predict(x)[0]

