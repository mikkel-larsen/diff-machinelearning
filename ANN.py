from abc import ABC
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler

tf.config.set_visible_devices([], 'GPU')

class create_diffANN(keras.Model, ABC):
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
        return self.model.predict(x)

    def compile(self, mean=0., sd=1., w=None, n_hidden=4, n_neurons=6, act='softplus', lr=0.1, seed=None, **kwargs):
        self.model = create_diffANN(mean, sd, n_hidden, n_neurons, act, seed)
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
        scaler = StandardScaler()
        scaler.fit(x)
        y, *z = y
        y = np.array(y).reshape(-1, 1)
        z = np.array(z).reshape(-1, 1)
        w = None
        if z.size > 0:
            z_perm = np.transpose(z[:, :, np.newaxis], axes=(1, 0, 2))
            lmbda = np.dot(np.transpose(y), y) / np.dot(np.transpose(z_perm[0, :, :]), z_perm[0, :, :])
            w = lmbda / (1 + lmbda)

        self.compile(scaler.mean_, scaler.scale_, w, n_hidden, n_neurons, act, lr, seed, **kwargs)
        self.fit(x, y, epochs, batch_size, early_stopping, **kwargs)
        return self.model

    def predict(self, x):
        x = np.array(x).reshape(-1, 1)
        return self.model.predict(x)

    def transfer_weights(self, model):
        self.model.set_weights(model.model.get_weights())
        return self.model


###### Quick and dirty functions

def create_and_fit_diffANN(x, y, D, epochs=60, batch_size=200, nodes=6, activations="softplus", seed=None):
    scaler = StandardScaler()
    scaler.fit(x)
    D_perm = np.transpose(D[:, :, np.newaxis], axes=(1, 0, 2))
    lmbda = np.dot(np.transpose(y), y) / np.dot(np.transpose(D_perm[0, :, :]), D_perm[0, :, :])
    w = lmbda / (1 + lmbda)

    model = create_diffANN(scaler.mean_, scaler.scale_, 4, nodes, activations, seed)

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

    model = create_ANN(scaler.mean_, scaler.scale_, 4, nodes, activations, seed)

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


#################################################################################################
######################################## GRID SEARCH ############################################
#################################################################################################

'''
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal

from models import simulate_data, BlackScholes_eulerScheme, BlackScholes_helper
from options import Call
import time
import os
from tensorboard.plugins.hparams import api as hp

tf.config.set_visible_devices([], 'GPU')

# Set parameters
n = 200  # Number of samples
strike = 100  # Strike of basket
rf_rate = 0.0  # Risk-free rate
T = 0.3  # Time-to-maturity of basket call
spot_rng = [50, 150]
vol = 0.2  # Volatility in the model

option = Call(strike, T)
model = BlackScholes_eulerScheme(rf_rate, vol ** 2)
helper = BlackScholes_helper(rf_rate, vol)

# Create data
x, y, z = simulate_data(n, spot_rng, option, model)
x_val = np.linspace(50, 150, 100).reshape(-1, 1)
y_val = helper.call_price(x_val, strike, T).reshape(-1, 1)
z_val = helper.call_delta(x_val, strike, T).reshape(-1, 1)

# Create tensorboard
root_logdir = os.path.join(os.curdir, "logs")


def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

# Create hyperparameter space
HP_NUM_NEURONS = hp.HParam('n_neurons', hp.Discrete([2, 4, 6, 8, 10]))
HP_LR = hp.HParam('lr', hp.Discrete([0.001, 0.01, 0.1, 1.]))
HP_NUM_HIDDEN_LAYERS = hp.HParam('n_hidden', hp.Discrete([1, 3, 5, 7, 9]))
#HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([50, 100, 150, 200]))

METRIC_MSE = 'mse'
METRIC_LOSS = 'loss'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_NUM_NEURONS, HP_LR, HP_NUM_HIDDEN_LAYERS],
        metrics=[hp.Metric(METRIC_MSE, display_name='MSE'), hp.Metric(METRIC_LOSS, display_name='Loss')],
    )


# Wrapper
def build_model(hparams):
    nn = NeuralNetwork()
    nn.compile(mean=100, sd=28.5, w=0.999,
               n_hidden=hparams[HP_NUM_HIDDEN_LAYERS],
               n_neurons=hparams[HP_NUM_NEURONS],
               lr=hparams[HP_LR],
               metrics=['mse'])
    nn.model.fit(x, [y, z],
                 epochs=10,
                 batch_size=30,
                 verbose=0)
    loss, loss1, loss2, mse1, mse2 = nn.model.evaluate(x_val, y_val)
    return loss, mse1


def run(run_logdir, hparams):
  with tf.summary.create_file_writer(run_logdir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    loss, mse = build_model(hparams)
    tf.summary.scalar(METRIC_MSE, mse, step=1)
    tf.summary.scalar(METRIC_LOSS, loss, step=1)

# Run and log
session_num = 0
for n_neurons in HP_NUM_NEURONS.domain.values:
    for lr in HP_LR.domain.values:
        for n_hidden in HP_NUM_HIDDEN_LAYERS.domain.values:
            hparams = {
                HP_NUM_NEURONS: n_neurons,
                HP_LR: lr,
                HP_NUM_HIDDEN_LAYERS: n_hidden,
            }
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            run(run_logdir + run_name, hparams)
            session_num += 1

'''



'''
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
'''
