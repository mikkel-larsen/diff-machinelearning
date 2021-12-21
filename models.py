import numpy as np
from numpy import random
from scipy.stats import norm

class Bachelier:
    def __init__(self, rf_rate=0, vol=None):
        self.rf_rate = rf_rate
        self.vol = vol

    def price(self, spot, strike, T, vol=None, rf_rate=None):
        if vol is None:
            vol = self.vol
        if rf_rate is None:
            rf_rate = self.rf_rate

        if rf_rate == 0:
            diff = spot - strike
            const = vol * np.sqrt(T)
            return diff * norm.cdf(diff / const) + const * norm.pdf(diff / const)

        Kstar = strike * np.exp(-rf_rate * T)
        v = vol * np.sqrt((1 - np.exp(-2 * rf_rate * T)) / (2 * rf_rate))
        diff = (spot - Kstar)
        return diff * norm.cdf(diff / v) + v * norm.pdf(diff / v)

    @staticmethod
    def simulate_basket(n, m, min, max, strike, T, w, cov):
        shape = [n, m]  # n simulations of m stocks
        s0 = random.uniform(min, max, shape)  # initialize with uniform dist.
        b0 = np.dot(s0, w)  # initial basket value
        s0 = s0[np.argsort(b0)]
        b0 = np.sort(b0).reshape(-1, 1)

        sT = np.array(s0)
        mean = np.zeros(m)
        sT += np.sqrt(T) * random.multivariate_normal(mean, cov, n)  # Simulated endpoint for all stocks
        bT = np.dot(sT, w)  # n simulated basket values at time T

        payoff = np.maximum(bT - strike, 0).reshape(-1, 1)  # payoff of european call option

        Z = np.dot(np.where(bT > strike, 1, 0).reshape(-1, 1), w.reshape(1, -1))  # Pathwise differentials
        Z = np.transpose(Z[:, :, np.newaxis], axes=(1, 0, 2))  # transpose so dimensions fit

        return s0, b0, payoff, Z

w = np.array([0.5, 0.5])
cov = np.array([[200, 0], [0, 200]])
x, b, y, Z = Bachelier.simulate_basket(5, 2, 0, 10, 5, 1, w, cov)
print(x)
print(b)

class BlackScholes:
    def __init__(self, rf_rate, vol):
        self.rf_rate = rf_rate
        self.vol = vol

    def price(self, spot, strike, T):
        d1 = (np.log(spot / strike) + (self.rf_rate + 0.5 * self.vol ** 2) * T) / (self.vol * np.sqrt(T))
        d2 = d1 - self.vol * np.sqrt(T)
        return spot * norm.cdf(d1) - strike * np.exp(-self.rf_rate * T) * norm.cdf(d2)

    def simulate_path(self, spot, T):
        dt = T / 252
        n = len(spot)
        z = random.standard_normal((n, int(1 / dt)))
        ret = np.zeros((n, int(1 / dt)), dtype=float)
        ret[:, 0] = spot
        for i in range(1, int(1 / dt)):
            ret[:, i] = ret[:, i - 1] * (1 + self.rf_rate * dt * + self.vol * np.sqrt(dt) * z[:, i])
        return ret

    def simulate_endpoint(self, spot, T):
        z = random.standard_normal(np.shape(spot))
        return np.array(spot * np.exp((self.rf_rate - self.vol ** 2 / 2) * T + self.vol * np.sqrt(T) * z))

    def simulate_data(self, n, min, max, strike, T, vol=None, rf_rate=None):
        if vol is None:
            vol = self.vol
        if rf_rate is None:
            rf_rate = self.rf_rate

        s0 = np.sort(random.uniform(min, max, n)).reshape(-1, 1)
        z = random.standard_normal(n).reshape(-1, 1)
        sT = s0 * np.exp((self.rf_rate - self.vol ** 2 / 2) * T + self.vol * np.sqrt(T) * z)

        payoff = np.maximum(sT - strike, 0)
        Z = np.where(sT > strike, 1, 0) * sT / s0

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

