import numpy as np
from numpy import random
from scipy.stats import norm
from scipy.stats import multivariate_normal
import tensorflow as tf


class EulerScheme:  # Abstract base class for different models
    def cache_dt(self):
        # Initialize stuff that do not need to be
        pass  # calculated in every iteration of euler scheme

    def update_price_path(self, *args):
        raise NotImplementedError  # A single euler step

class Bachelier_eulerScheme(EulerScheme):
    def __init__(self, rf_rate, vol):
        self.const = None
        self.dt = None
        self.dt_sqrt = None
        self.rf_rate = rf_rate
        self.vol = vol

    def cache_dt(self, dt):
        self.dt = dt
        self.dt_sqrt = np.sqrt(dt)
        self.const = 1 + self.rf_rate * self.dt

    def update_price_path(self, s0, *args):
        shape = np.shape(s0)
        if type(self.vol) is not list and type(self.vol) is not np.ndarray:
            z = multivariate_normal(np.zeros(shape[1]), self.vol ** 2).rvs(size=shape[0]).reshape(shape)
        else:
            z = multivariate_normal(np.zeros(shape[1]), self.vol).rvs(size=shape[0]).reshape(shape)

        return s0 * self.const + self.dt_sqrt * z

class BlackScholes_eulerScheme(EulerScheme):
    def __init__(self, rf_rate, vol):
        self.const = None
        self.dt = None
        self.dt_sqrt = None
        self.rf_rate = rf_rate
        self.vol = vol

    def cache_dt(self, dt):
        self.dt = dt
        self.dt_sqrt = np.sqrt(dt)
        self.const = 1 + self.rf_rate * self.dt

    def simulate_endpoint(self, s0, T):
        shape = np.shape(s0)
        if type(self.vol) is not list and type(self.vol) is not np.ndarray:
            var = self.vol ** 2
            z = multivariate_normal(np.zeros(shape[1]), var).rvs(size=shape[0]).reshape(shape)
        else:
            var = np.diag(self.vol)
            z = multivariate_normal(np.zeros(shape[1]), self.vol).rvs(size=shape[0]).reshape(shape)

        return s0 * np.exp((self.rf_rate - var / 2.0) * T + np.sqrt(T) * z)

    def update_price_path(self, s0, *args):
        shape = np.shape(s0)
        if type(self.vol) is not list and type(self.vol) is not np.ndarray:
            z = multivariate_normal(np.zeros(shape[1]), self.vol ** 2).rvs(size=shape[0]).reshape(shape)
        else:
            z = multivariate_normal(np.zeros(shape[1]), self.vol).rvs(size=shape[0]).reshape(shape)

        return s0 * (self.const + self.dt_sqrt * z)

def simulate_data(n, rng, option, model, seed=None):
    if seed is not None:
        tf.random.set_seed(seed)  # Possibility of setting seed for reproducibility

    if type(n) is not list and type(n) is not np.ndarray:
        n = [n, 1]  # Failsafe: random number generators takes lists as input

    if np.size(rng) == 2:  # Randomly generate spots in given range
        spot_min, spot_max = rng
    else:
        spot_min = rng  # If only a single number is given all spots will start here
        spot_max = rng

    is_simple = False  # Check if derivative is simple (and vocal about it)
    if hasattr(option, 'is_simple'):
        if option.is_simple:
            is_simple = True

    T = option.T

    if is_simple and hasattr(model, 'simulate_endpoint'):  # Skip euler steps if possible
        with tf.GradientTape() as tape:
            # Randomly select spots
            # s0 = tf.sort(tf.random.uniform(n, spot_min, spot_max, dtype=tf.float64), axis=0)
            s0 = tf.random.uniform(n, spot_min, spot_max, dtype=tf.float64)
            tape.watch(s0)  # Spots are not tf.Variables, we tell tf to track anyway
            sT = model.simulate_endpoint(s0, T)
            payoff = option.payoff(sT)
        Z = tape.gradient(payoff, s0)
    else:
        if T < 1:
            end = 252  # Update 252 times
            dt = T / 252
        else:
            end = int(T * 252)  # Update once every day
            dt = 1 / 252

        # Initialize constants that do not need to be
        model.cache_dt(dt)  # recalculated for every iteration in euler scheme update

        with tf.GradientTape() as tape:
            # Randomly select spots
            # s0 = tf.sort(tf.random.uniform(n, spot_min, spot_max, dtype=tf.float64), axis=0)
            s0 = tf.random.uniform(n, spot_min, spot_max, dtype=tf.float64)
            tape.watch(s0)  # Spots are not tf.Variables, we tell tf to track anyway
            st = s0
            price_path = [st]  # List to hold price-path
            for _ in range(end):
                st = model.update_price_path(st, dt)  # Update with single euler step
                price_path.append(st)  # Add to price-path

            if is_simple:  # Find payoff
                payoff = option.payoff(price_path[-1])
            else:
                payoff = option.payoff(price_path)

        # Get gradient of payoff w.r.t. spot (s0)
        Z = tape.gradient(payoff, s0)

        #s0 = tf.reshape(s0, n)
        payoff = tf.reshape(payoff, (n[0], -1))
        #Z = tf.reshape(Z, (shape[0], -1))

    return np.array(s0), np.array(payoff), np.array(Z)

# ------ HELPER FUNCTIONS ------

class Bachelier:
    def __init__(self, rf_rate=0.0, vol=None):
        self.rf_rate = rf_rate
        self.vol = vol

    def call_price(self, spot, strike, T, vol=None, rf_rate=None):
        if vol is None:
            vol = self.vol
        if rf_rate is None:
            rf_rate = self.rf_rate

        if rf_rate == 0:
            diff = spot - strike
            const = vol * np.sqrt(T)
            return diff * norm.cdf(diff / const) + const * norm.pdf(diff / const)

        D = (spot * np.exp(rf_rate * T) - strike) / (vol * np.sqrt(T))
        return np.exp(-rf_rate*T) * vol * np.sqrt(T) * (D * norm.cdf(D) + norm.pdf(D))

    def call_delta(self, spot, strike, T, vol=None, rf_rate=None):
        if vol is None:
            vol = self.vol
        if rf_rate is None:
            rf_rate = self.rf_rate

        d = (spot - strike) / vol / np.sqrt(T)
        return norm.cdf(d)

    def simulate_price_path(self, n, spot, T, n_yearly_updates=252):
        if np.size(spot) == 2:
            spot_min, spot_max = spot
        else:
            spot_min = spot
            spot_max = spot

        if T < 1:
            end = n_yearly_updates  # Update 252 times (default)
            dt = T / n_yearly_updates
        else:
            end = int(T * n_yearly_updates)  # Update once every day (default)
            dt = 1 / n_yearly_updates

        spots = np.zeros((n, end))
        spots[:, 0] = random.uniform(spot_min, spot_max, n)

        z = random.standard_normal((n, end))
        dt_sqrt = np.sqrt(dt)
        for i in range(1, end):
            spots[:, i] = spots[:, i - 1] * (1 + self.rf_rate * dt) + self.vol * dt_sqrt * z[:, i]

        return spots

    def simulate_basket(self, n, m, rng, option, w, cov, seed=None):
        if seed is not None:
            tf.random.set_seed(seed)
        if np.size(rng) == 2:
            spot_min, spot_max = rng
        else:
            spot_min = rng
            spot_max = rng

        with tf.GradientTape() as tape:
            T = option.T
            shape = [n, m]
            s0 = tf.random.uniform(shape, spot_min, spot_max, dtype=tf.float64)
            tape.watch(s0)
            b0 = s0 @ w.reshape(-1, 1)
            order = tf.reshape(tf.argsort(b0, axis=0), -1)
            s0 = tf.gather(s0, order)
            b0 = tf.sort(b0, axis=0)

            sT = s0
            mean = tf.zeros(m)
            if self.rf_rate == 0.0:
                sT += np.sqrt(T) * random.multivariate_normal(mean, cov, n)
            else:
                if T < 1:
                    end = tf.constant(252, dtype=tf.int64)  # Update 252 times
                    dt = tf.constant(T / 252, dtype=tf.float64)
                else:
                    end = tf.constant(T * 252, dtype=tf.int64)  # Update once every day times
                    dt = tf.constant(1 / 252, dtype=tf.float64)

                dt_sq = tf.sqrt(dt)
                for _ in tf.range(end):
                    z = random.multivariate_normal(mean, cov, n)
                    sT += sT * self.rf_rate * dt + dt_sq * z

            bT = sT @ w.reshape(-1, 1)

            payoff = option.payoff(bT)
            Z = tape.gradient(payoff, s0)
        '''

        shape = [n, m]  # n simulations of m stocks
        s0 = random.uniform(min, max, shape)  # initialize with uniform dist.
        b0 = np.dot(s0, w)  # initial basket value
        s0 = s0[np.argsort(b0)]
        b0 = np.sort(b0).reshape(-1, 1)

        sT = np.array(s0)
        mean = np.zeros(m)
        sT += np.sqrt(T) * random.multivariate_normal(mean, cov, n)  # Simulated endpoint for all stocks
        bT = np.dot(sT, w)  # n simulated basket values at time T

        payoff = np.maximum(bT - strike, 0).reshape(-1, 1)  # payoff of european option option

        Z = np.dot(np.where(bT > strike, 1, 0).reshape(-1, 1), w.reshape(1, -1))  # Pathwise differentials
        Z = np.transpose(Z[:, :, np.newaxis], axes=(1, 0, 2))  # transpose so dimensions fit
        '''
        return np.array(s0), np.array(b0), np.array(payoff), np.array(Z)


class BlackScholes:
    def __init__(self, rf_rate, vol):
        self.rf_rate = rf_rate
        self.vol = vol

    def call_price(self, spot, strike, T):
        d1 = (np.log(spot / strike) + (self.rf_rate + 0.5 * self.vol ** 2) * T) / (self.vol * np.sqrt(T))
        d2 = d1 - self.vol * np.sqrt(T)
        return spot * norm.cdf(d1) - strike * np.exp(-self.rf_rate * T) * norm.cdf(d2)

    def call_delta(self, spot, strike, T):
        d1 = (np.log(spot / strike) + (self.rf_rate + 0.5 * self.vol ** 2) * T) / (self.vol * np.sqrt(T))
        return norm.cdf(d1)

    def put_price(self, spot, strike, T):
        d1 = (np.log(spot / strike) + (self.rf_rate + 0.5 * self.vol ** 2) * T) / (self.vol * np.sqrt(T))
        d2 = d1 - self.vol * np.sqrt(T)
        return strike * np.exp(-self.rf_rate * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)

    def put_delta(self, spot, strike, T):
        d1 = (np.log(spot / strike) + (self.rf_rate + 0.5 * self.vol ** 2) * T) / (self.vol * np.sqrt(T))
        return norm.cdf(d1) - 1

    def straddle_price(self, spot, strike, T):
        return self.call_price(spot, strike, T) + self.put_price(spot, strike, T)

    def straddle_delta(self, spot, strike, T):
        return self.call_delta(spot, strike, T) + self.put_delta(spot, strike, T)

    def simulate_price_path(self, n, spot, T, n_yearly_updates=252):
        if np.size(spot) == 2:
            spot_min, spot_max = spot
        else:
            spot_min = spot
            spot_max = spot

        if T < 1:
            end = n_yearly_updates  # Update 252 times (default)
            dt = T / n_yearly_updates
        else:
            end = int(T * n_yearly_updates)  # Update once every day (default)
            dt = 1 / n_yearly_updates

        spots = np.zeros((n, end))
        spots[:, 0] = random.uniform(spot_min, spot_max, n)
        z = random.standard_normal((n, end))
        dt_sqrt = np.sqrt(dt)
        for i in range(1, end):
            spots[:, i] = spots[:, i - 1] * (1 + self.rf_rate * dt + self.vol * dt_sqrt * z[:, i])

        return spots

    def simulate_data(self, n, rng, option, seed=None):
        if seed is not None:
            tf.random.set_seed(seed)

        T = option.T

        if np.size(rng) == 2:
            spot_min, spot_max = rng
        else:
            spot_min = rng
            spot_max = rng

        simple = False
        if hasattr(option, 'is_simple'):
            if option.is_simple:
                simple = True

        if simple:
            with tf.GradientTape() as tape:
                s0 = tf.sort(tf.random.uniform([n, 1], spot_min, spot_max, dtype=tf.float64), axis=0)
                tape.watch(s0)
                z = tf.random.normal([n, 1], dtype=tf.float64)
                sT = s0 * tf.exp((self.rf_rate - self.vol ** 2 / 2.0) * tf.cast(T, tf.float64) +
                                 self.vol * tf.sqrt(tf.cast(T, tf.float64)) * z)
                payoff = option.payoff(sT)
            Z = tape.gradient(payoff, s0)
        else:
            if T < 1:
                end = 252  # Update 252 times (default)
                dt = T / 252
            else:
                end = int(T * 252)  # Update once every day (default)
                dt = 1 / 252

            with tf.GradientTape() as tape:
                s0 = tf.sort(tf.random.uniform([1, n], spot_min, spot_max, dtype=tf.float64), axis=0)
                tape.watch(s0)
                z = random.standard_normal((n, end))
                dt_sqrt = np.sqrt(dt)
                sT = s0
                price_path = [sT]
                for i in range(end):
                    sT += sT * (self.rf_rate * dt + self.vol * dt_sqrt * z[:, i])
                    price_path.append(sT)

                payoff = option.payoff(price_path)
            Z = tape.gradient(payoff, s0)
            s0 = tf.reshape(s0, (-1, 1))
            payoff = tf.reshape(payoff, (-1, 1))
            Z = tf.reshape(Z, (-1, 1))

        return np.array(s0), np.array(payoff), np.array(Z)

    @staticmethod
    def simulate(n, min, max, option, vol, rf_rate, seed=None):
        if seed is not None:
            tf.random.set_seed(seed)

        T = option.T

        with tf.GradientTape() as tape:
            s0 = tf.sort(tf.random.uniform([n, 1], min, max, dtype=tf.float64), axis=0)
            tape.watch(s0)
            z = tf.random.normal([n, 1], dtype=tf.float64)
            sT = s0 * tf.exp((rf_rate - vol ** 2 / 2.0) * tf.cast(T, tf.float64) + vol * tf.sqrt(tf.cast(T, tf.float64)) * z)
            payoff = option.payoff(sT)

        Z = tape.gradient(payoff, s0)

        return s0, payoff, Z


def transform(x, K, b=None):
    s_min = np.min(x)
    s_max = np.max(x)
    if b is None:
        b = 15 / (s_max - s_min)
    x = (x - s_min) / (s_max - s_min)
    c1 = np.arcsinh(b * (s_min - K))
    c2 = np.arcsinh(b * (s_max - K))
    return np.sinh(c2 * x + c1 * (1 - x)) / b + K

