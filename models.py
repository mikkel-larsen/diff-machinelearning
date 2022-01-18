import numpy as np
from numpy import random
from scipy.stats import norm
import tensorflow as tf

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

    @staticmethod
    def simulate_basket(n, m, min, max, option, w, cov, rf_rate=0.0):
        with tf.GradientTape() as tape:
            T = option.T
            shape = [n, m]
            # s0 = tf.constant(tf.random.uniform(shape, min, max, dtype=tf.float64))
            s0 = tf.constant(random.uniform(min, max, shape), dtype=tf.float64)
            tape.watch(s0)
            b0 = s0 @ w.reshape(-1, 1)
            order = tf.reshape(tf.argsort(b0, axis=0), -1)
            s0 = tf.gather(s0, order)
            b0 = tf.sort(b0, axis=0)

            sT = s0
            mean = tf.zeros(m)
            if rf_rate == 0.0:
                sT += np.sqrt(T) * random.multivariate_normal(mean, cov, n)
            else:
                dt = tf.constant(T / 252, dtype=tf.float64)
                dt_sq = tf.sqrt(dt)
                # z = random.multivariate_normal(mean, cov, 252)
                for i in tf.range(int(252)):
                    z = random.multivariate_normal(mean, cov, n)
                    sT += sT * rf_rate * dt + dt_sq * z

            bT = sT @ w.reshape(-1, 1)

            # payoff = tf.maximum(bT - strike, 0.0)
            payoff = option.payoff(bT)
            Z = tape.gradient(payoff, s0)
            Z = tf.transpose(Z[:, :, np.newaxis], perm=(1, 0, 2))
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

    def simulate_data(self, n, min, max, option, vol=None, rf_rate=None):
        if vol is None:
            vol = self.vol
        if rf_rate is None:
            rf_rate = self.rf_rate

        T = option.T

        with tf.GradientTape() as tape:
            s0 = tf.sort(tf.random.uniform([n, 1], min, max, dtype=tf.float64), axis=0)
            tape.watch(s0)
            z = tf.random.normal([n, 1], dtype=tf.float64)
            sT = s0 * tf.exp((rf_rate - vol ** 2 / 2.0) * tf.cast(T, tf.float64) + vol * tf.sqrt(tf.cast(T, tf.float64)) * z)
            payoff = option.payoff(sT)

        Z = tape.gradient(payoff, s0)
        '''
        s0 = np.sort(random.uniform(min, max, n)).reshape(-1, 1)
        z = random.standard_normal(n).reshape(-1, 1)
        sT = s0 * np.exp((rf_rate - vol ** 2 / 2) * T + vol * np.sqrt(T) * z)

        payoff = np.maximum(sT - strike, 0)
        Z = np.where(sT > strike, 1, 0) * sT / s0
        '''
        return np.array(s0), np.array(payoff), np.array(Z)

    @staticmethod
    def simulate(n, min, max, option, vol, rf_rate):
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

