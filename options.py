import tensorflow as tf
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class Digital:
    def __init__(self, strike, timetomat, epsilon):
        self.strike = strike
        self.T = timetomat
        self.epsilon = epsilon
        self.a = tf.constant(1 / (2 * epsilon), dtype=tf.float32) if epsilon > 0 else 0
        self.b = tf.constant((epsilon - strike) / (2 * epsilon), dtype=tf.float32) if epsilon > 0 else 0
        self.is_simple = True

    def payoff(self, spots, **kwargs):
        if self.epsilon == 0:
            return tf.where(spots >= self.strike, 1, 0)
        return tf.minimum(1, tf.maximum(0, self.a * spots + self.b))


class Linear:
    def __init__(self, timetomat):
        self.T = timetomat
        self.is_simple = False

    def payoff(self, spots, **kwargs):
        return spots[-1]


class Spread:
    def __init__(self, timetomat):
        self.T = timetomat
        self.is_simple = True

    def payoff(self, spots, **kwargs):
        return tf.abs(spots[:, 0] - spots[:, 1])


class Bond_option:
    def __init__(self, strike, T, S, term_structure):
        self.strike = strike
        self.T = T
        self.S = S
        self.term_structure = term_structure
        self.is_simple = False

    def payoff(self, rates, **kwargs):
        dt = kwargs['dt']
        discounting = tf.exp(-tf.math.reduce_sum(rates, axis=0) * dt)

        return tf.maximum(discounting * (self.term_structure(self.T, self.S, rates[-1]) - self.strike), 0)


class Zero_Coupon_Bond:
    def __init__(self, timetomat):
        self.T = timetomat

    def payoff(self, rate_path, **kwargs):
        dt = kwargs['dt']
        return tf.exp(-tf.reduce_sum(rate_path, axis=0) * dt)


class Swaption:
    def __init__(self, strike, timetomat, settlement_dates, term_structure):
        self.strike = strike
        self.coverage = settlement_dates[1] - settlement_dates[0]
        self.T = timetomat
        self.settlement_dates = settlement_dates
        self.term_structure = term_structure
        self.is_simple = False

    def payoff(self, rate, **kwargs):
        dt = kwargs['dt']
        discounting = tf.exp(-tf.math.reduce_sum(rate, axis=0) * dt)

        sum_discountings = 0
        for d in self.settlement_dates:
            sum_discountings += self.term_structure(self.T, d, rate[-1])

        par_rate = (1 - self.term_structure(self.T, self.settlement_dates[-1], rate[-1])) / \
                   (self.coverage * sum_discountings)

        pv = 0
        for d in self.settlement_dates:
            pv += (self.strike - par_rate) * self.term_structure(self.T, d, rate[-1]) * self.coverage

        return tf.maximum(discounting * pv, 0)

class Call:
    def __init__(self, strike, timetomat):
        self.strike = strike
        self.T = timetomat
        self.is_simple = False

    def payoff(self, spot, **kwargs):
        return tf.maximum(spot[-1] - self.strike, 0)


class Call_geometric:
    def __init__(self, strike, T):
        self.T = T
        self.strike = strike

    def payoff(self, spot, **kwargs):
        return tf.maximum(tf.exp(tf.reduce_mean(tf.math.log(spot), axis=0)) - self.strike, 0)


class Call_basket:
    def __init__(self, strike, timetomat, w):
        self.strike = strike
        self.T = timetomat
        self.w = w
        self.is_simple = True

    def payoff(self, spot, **kwargs):
        return tf.maximum(tf.reduce_sum(spot * self.w, axis=1) - self.strike, 0)


class Put:
    def __init__(self, strike, timetomat):
        self.strike = strike
        self.T = timetomat
        self.is_simple = True

    def payoff(self, spot, **kwargs):
        return tf.maximum(self.strike - spot, 0)


class Straddle:
    def __init__(self, strike, timetomat):
        self.strike = strike
        self.T = timetomat
        self.is_simple = True

    def payoff(self, spot, **kwargs):
        return tf.maximum(self.strike - spot, spot - self.strike)
