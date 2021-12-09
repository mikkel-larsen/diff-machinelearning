import matplotlib.pyplot as plt
import numpy as np
from numpy import random

class BS:
    def __init__(self, rf_rate, vol, T):
        self.rf_rate = rf_rate
        self.vol = vol
        self.T = T

    def BS_simulate_path(self, m, spot):
        dt = self.T / m
        n = len(spot)
        z = random.standard_normal((n, int(1 / dt)))
        ret = np.zeros((n, int(1 / dt)), dtype=float)
        ret[:, 0] = spot
        for i in range(1, int(1 / dt)):
            ret[:, i] = ret[:, i - 1] * (1 + self.rf_rate * dt * + self.vol * np.sqrt(dt) * z[:, i])
        return ret

    def BS_simulate_endpoint(self, n, spot):
        z = random.standard_normal(n)
        return spot * np.exp((self.rf_rate - self.vol ** 2 / 2) * self.T + self.vol * np.sqrt(self.T) * z)


def transform(x, K, b = None):
    s_min = np.min(x)
    s_max = np.max(x)
    if b == None:
        b = 15 / (s_max - s_min)
    x = (x - s_min) / (s_max - s_min)
    c1 = np.arcsinh(b * (s_min - K))
    c2 = np.arcsinh(b * (s_max - K))
    return np.sinh(c2 * x + c1 * (1 - x)) / b + K