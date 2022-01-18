import tensorflow as tf

class Call:
    def __init__(self, strike, timetomat):
        self.strike = strike
        self.T = timetomat

    @tf.function
    def payoff(self, spot):
        return tf.maximum(spot - self.strike, 0)

class Put:
    def __init__(self, strike, timetomat):
        self.strike = strike
        self.T = timetomat

    @tf.function
    def payoff(self, spot):
        return tf.maximum(self.strike - spot, 0)

class Straddle:
    def __init__(self, strike, timetomat):
        self.strike = strike
        self.T = timetomat

    @tf.function
    def payoff(self, spot):
        return tf.maximum(self.strike - spot, spot - self.strike)
