import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from scipy.stats import norm

class Basket:
    def __init__(self, n, w, strike):
        self.n = n
        self.w = w
        self.strike = strike

    def __str__(self):
        return "{}".format(self.spots)

    def make_basket_uniform(self, min, max, m):
        self.spots = np.random.uniform(min, max, (m, self.n))
        return self.spots

class Bachelier:
    def __init__(self, rf_rate, vol):
        self.rf_rate = rf_rate
        self.vol = vol

    def price_basket(self, basket, cov, T):
        spot = np.dot(basket.spots, basket.w)
        vol = np.sqrt(np.dot(np.transpose(basket.w), np.dot(cov, basket.w)))
        if self.rf_rate == 0.0:
            diff = spot - basket.strike
            const = vol * np.sqrt(T)
            return diff * norm.cdf(diff / const) + const * norm.pdf(diff / const)

        Kstar = basket.strike * np.exp(-self.rf_rate * T)
        v = vol * np.sqrt((1 - np.exp(-2 * self.rf_rate * T)) / (2 * self.rf_rate))
        diff = (spot - Kstar)
        return diff * norm.cdf(diff / v) + v * norm.pdf(diff / v)

    def price(self, spot, strike, T):
        if self.rf_rate == 0:
            diff = spot - strike
            const = self.vol * np.sqrt(T)
            return diff * norm.cdf(diff / const) + const * norm.pdf(diff / const)

        Kstar = strike * np.exp(-self.rf_rate * T)
        v = self.vol * np.sqrt((1 - np.exp(-2 * self.rf_rate * T)) / (2 * self.rf_rate))
        diff = (spot - Kstar)
        return diff * norm.cdf(diff / v) + v * norm.pdf(diff / v)

    def simulate_basket_endpoint(self, basket, cov, T):
        ret = np.array(basket.spots, dtype='float')
        shape = list(np.shape(basket.spots))
        mean = np.zeros(shape[1])
        if self.rf_rate == 0:
            return ret + np.sqrt(T) * random.multivariate_normal(mean, cov, shape[0])

        dt = 1 / 252
        for i in range(int(T * 252)):
            z = random.multivariate_normal(mean, cov, shape[0])
            ret += ret * self.rf_rate * dt + np.sqrt(dt) * z

        return ret
    '''
    def simulate_endpoint(self, spot, T):
        dt = 1 / (T * 252)
        shape = list(np.shape(spot))
        shape.insert(0, int(T * 252))
        z = random.standard_normal(shape)
        ret = np.array(spot, dtype='float')
        if len(shape) == 3:
            for i in range(int(T * 252)):
                ret += ret * self.rf_rate * dt + self.vol * np.sqrt(dt) * z[i, :, :]
        if len(shape) == 2:
            for i in range(int(T * 252)):
                ret += (ret * self.rf_rate * dt + self.vol * np.sqrt(dt) * z[i, :])
        return ret
    '''

class BlackScholes:
    def __init__(self, rf_rate, vol):
        self.rf_rate = rf_rate
        self.vol = vol

    def price(self, spot, strike, T):
        d1 = (np.log(spot / strike) + (self.rf_rate + 0.5 * self.vol ** 2) * T) / (self.vol * np.sqrt(T))
        d2 = d1 - self.vol * np.sqrt(T)
        return spot * norm.cdf(d1) - strike * np.exp(-self.rf_rate * self.T) * norm.cdf(d2)

    def simulate_path(self, spot, T):
        dt = T / 252
        n = len(spot)
        z = random.standard_normal((n, int(1 / dt)))
        ret = np.zeros((n, int(1 / dt)), dtype=float)
        ret[:, 0] = spot
        for i in range(1, int(1 / dt)):
            ret[:, i] = ret[:, i - 1] * (1 + self.rf_rate * dt * + self.vol * np.sqrt(dt) * z[:, i])
        return ret

    def simulate_endpoint(self, n, spot, T):
        z = random.standard_normal((len(spot), n))
        return np.array(spot * np.exp((self.rf_rate - self.vol ** 2 / 2) * T + self.vol * np.sqrt(T) * z))


def transform(x, K, b = None):
    s_min = np.min(x)
    s_max = np.max(x)
    if b == None:
        b = 15 / (s_max - s_min)
    x = (x - s_min) / (s_max - s_min)
    c1 = np.arcsinh(b * (s_min - K))
    c2 = np.arcsinh(b * (s_max - K))
    return np.sinh(c2 * x + c1 * (1 - x)) / b + K

