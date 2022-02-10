import tensorflow as tf
import numpy as np

class Swaption:
    def __init__(self, strike, timetomat, settlement_dates, swap_price):
        self.strike = strike
        self.T = timetomat
        self.settlement_dates = settlement_dates
        self.swap_price = swap_price
        self.dt = None
        self.is_simple = True

    def initialize(self, **kwargs):
        self.dt = kwargs['dt']

    def payoff(self, rate):
        return tf.maximum(self.swap_price(rate) - self.strike, 0)


class Swap:
    def __init__(self, strike, timetomat, settlement_dates):
        self.strike = strike
        self.T = timetomat
        self.dates = settlement_dates
        self.dt = None

    def initialize(self, **kwargs):
        self.dt = kwargs['dt']

    def payoff(self, price_path):
        running_sum = 0
        for d in self.dates:
            running_sum += (price_path[d] - self.strike) * np.exp(-np.sum(np.array(price_path[0:d]) * self.dt))
        return running_sum


class Call:
    def __init__(self, strike, timetomat):
        self.strike = strike
        self.T = timetomat
        self.is_simple = True

    @tf.function
    def payoff(self, spot):
        return tf.maximum(spot - self.strike, 0)


class Call_geometric:
    def __init__(self, strike, T):
        self.T = T
        self.strike = strike

    @tf.function
    def payoff(self, spot):
        return tf.maximum(tf.exp(tf.reduce_mean(tf.math.log(spot), axis=0)) - self.strike, 0)


class Call_basket:
    def __init__(self, strike, timetomat, w):
        self.strike = strike
        self.T = timetomat
        self.w = w
        self.is_simple = True

    @tf.function
    def payoff(self, spot):
        return tf.maximum(tf.reduce_sum(spot * self.w, axis=1) - self.strike, 0)


class Put:
    def __init__(self, strike, timetomat):
        self.strike = strike
        self.T = timetomat
        self.is_simple = True

    @tf.function
    def payoff(self, spot):
        return tf.maximum(self.strike - spot, 0)


class Straddle:
    def __init__(self, strike, timetomat):
        self.strike = strike
        self.T = timetomat
        self.is_simple = True

    @tf.function
    def payoff(self, spot):
        return tf.maximum(self.strike - spot, spot - self.strike)
