import tensorflow as tf
import numpy as np

class Call:
    def __init__(self, strike, timetomat):
        self.strike = strike
        self.T = timetomat
        self.is_simple = True

    @tf.function
    def payoff(self, spot):
        return tf.maximum(spot - self.strike, 0)

class Call_basket:
    def __init__(self, strike, timetomat, w):
        self.strike = strike
        self.T = timetomat
        self.w = w
        self.is_simple = True

    @tf.function
    def payoff(self, spot):
        return tf.maximum(tf.reduce_sum(spot * self.w, 1) - self.strike, 0)

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
