import numpy as np
import tensorflow as tf

class Call:
    def __init__(self, strike, timetomat):
        self.strike = strike
        self.T = timetomat

    @tf.function
    def payoff(self, spot):
        return tf.maximum(spot - self.strike, 0)
