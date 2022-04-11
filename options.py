import tensorflow as tf
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from models import Vasicek_helper, BlackScholes_helper
from ANN import create_and_fit_ANN

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
    def __init__(self, strike, timetomat, settlement_dates, term_structure, opt_type='receiver'):
        self.strike = strike
        self.coverage = settlement_dates[1] - settlement_dates[0]
        self.T = timetomat
        self.settlement_dates = settlement_dates
        self.term_structure = term_structure
        self.type = opt_type
        self.is_simple = False

    def payoff(self, rate, **kwargs):
        dt = kwargs['dt']
        shape = np.shape(np.squeeze(rate))
        axis = np.delete(np.arange(len(shape)), 1)
        discounting = tf.exp(-tf.math.reduce_sum(rate, axis=axis) * dt)

        sum_discountings = 0
        for d in self.settlement_dates:
            sum_discountings += self.term_structure(self.T, d, rate[-1])

        par_rate = (1 - self.term_structure(self.T, self.settlement_dates[-1], rate[-1])) / \
                   (self.coverage * sum_discountings)

        pv = 0
        for d in self.settlement_dates:
            pv += (self.strike - par_rate) * self.term_structure(self.T, d, rate[-1]) * self.coverage

        if self.type == 'payer':
            return tf.maximum(- discounting * pv, 0)

        return tf.maximum(discounting * pv, 0)


class Bermudan_Swaption:
    def __init__(self, strike, early_excer, settlement_dates, term_structure, opt_type='receiver'):
        self.strike = strike
        self.coverage = settlement_dates[1] - settlement_dates[0]
        self.exercise_dates = early_excer
        self.T = max(early_excer)
        self.settlement_dates = settlement_dates
        self.term_structure = term_structure
        self.type = opt_type
        #self.decision_boundaries = decision_boundaries
        self.is_simple = False

    def payoff(self, rate, **kwargs):
        dt = kwargs['dt']
        n = kwargs['n']
        n_updates = kwargs['end']
        exercise_dates_index = np.array([n_updates * t / self.T for t in self.exercise_dates], dtype=int)

        discounting = 1
        #pv = 0
        ret = []
        for i in range(len(self.exercise_dates)):
            #discounting = tf.exp(-tf.math.reduce_sum(rate[:exercise_dates_index[i]], axis=0) * dt)

            sum_discountings = 0
            for d in self.settlement_dates:
                sum_discountings += self.term_structure(self.exercise_dates[i], d, rate[exercise_dates_index[i]])

            par_rate = (1 - self.term_structure(self.exercise_dates[i], self.settlement_dates[-1], rate[exercise_dates_index[i]])) / \
                       (self.coverage * sum_discountings)
            pv = 0
            for d in self.settlement_dates:
                pv += (self.strike - par_rate) * self.term_structure(self.exercise_dates[i], d, rate[exercise_dates_index[i]]) * self.coverage

            if self.type == 'payer':
                pv = -pv
            ret.append(tf.maximum(discounting * pv, 0))

        pv1 = ret[-1]
        for i in range(1, len(self.exercise_dates)):
            pv0 = ret[-(i+1)]
            pv1 = pv1 * tf.exp(-tf.math.reduce_sum(rate[exercise_dates_index[-(i+1)]:exercise_dates_index[-i]], axis=0) * dt)

            continuation_model = create_and_fit_ANN(rate[exercise_dates_index[-(i+1)]], pv1)
            continuation_value = continuation_model(rate[exercise_dates_index[-(i+1)]])
            continuation_value = tf.cast(tf.maximum(continuation_value, 0.0), tf.float32)

            pv1 = tf.where(pv0 > continuation_value, pv0, pv1)

        return pv1 * tf.exp(-tf.math.reduce_sum(rate[:exercise_dates_index[0]], axis=0) * dt)


class Call:
    def __init__(self, strike, timetomat):
        self.strike = strike
        self.T = timetomat
        self.is_simple = True

    def payoff(self, spot, **kwargs):
        return tf.maximum(spot - self.strike, 0)

class Bermudan_Call:
    def __init__(self, strike, exersice_dates, p):
        self.strike = strike
        self.exersice_dates = exersice_dates
        self.T = max(exersice_dates)
        self.p = p
        self.is_simple = False

    def payoff(self, spot, **kwargs):
        n_updates = kwargs['end']
        exersice_dates_index = np.array([n_updates * t / self.T for t in self.exersice_dates], dtype=int)
        ret = []
        for i in range(len(self.exersice_dates)):
            ret.append(tf.maximum(spot[exersice_dates_index[i]] - self.strike, 0))

        pv0 = ret[0]
        pv1 = ret[1] * self.p(self.exersice_dates[1] - self.exersice_dates[0])
        helper = BlackScholes_helper(-0.1, 0.2)
        continuation_value = helper.call_price(spot[exersice_dates_index[0]], 100, self.exersice_dates[1] - self.exersice_dates[0])
        optimal_choice = tf.where(pv0 > continuation_value, pv0, pv1) * self.p(self.exersice_dates[0])
        return optimal_choice

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
