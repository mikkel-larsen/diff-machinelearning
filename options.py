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


class Bermudan_Swaption:
    def __init__(self, strike, early_excer, settlement_dates, term_structure):
        self.strike = strike
        self.coverage = settlement_dates[1] - settlement_dates[0]
        self.exercise_dates = early_excer
        self.T = max(early_excer)
        self.settlement_dates = settlement_dates
        self.term_structure = term_structure
        self.is_simple = False

    def payoff(self, rate, **kwargs):
        dt = kwargs['dt']
        n = kwargs['n']
        n_updates = kwargs['end']
        # T1_index = int(T1 / T2 * n_updates)
        # T2_index = n_updates
        # T_index = np.array([T1_index, T2_index])
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

            ret.append(tf.maximum(discounting * pv, 0))

        ret = tf.squeeze(tf.transpose(ret))
        #ret[:, 1] = ret[:, 1] * tf.exp(-tf.math.reduce_sum(rate[exercise_dates_index[0]:exercise_dates_index[1]], axis=0) * dt)
        pv1 = tf.reshape(ret[:, -1], (-1, 1))
        for i in range(len(self.exercise_dates) - 1):
            pv0 = tf.reshape(ret[:, -(i+2)], (-1, 1))
            pv1 *= tf.exp(-tf.math.reduce_sum(rate[exercise_dates_index[-(i+2)]:exercise_dates_index[-(i+1)]], axis=0) * dt)
            index = tf.squeeze(tf.where(tf.greater(tf.squeeze(pv0), 0)))
            x = tf.gather(pv0, index)
            y = tf.gather(pv1, index)

            design = PolynomialFeatures(degree=4)
            x = design.fit_transform(x)
            lin_model = LinearRegression(fit_intercept=False)
            lin_model.fit(x, y)
            estimates = lin_model.predict(design.transform(tf.reshape(pv0, (-1, 1))))
            pv1 = tf.where(tf.maximum(tf.cast(estimates, dtype=tf.float32), 0) >= pv0, pv1, pv0)

        return pv1 * tf.exp(-tf.math.reduce_sum(rate[:exercise_dates_index[0]], axis=0) * dt)


        '''
        ret_copy = np.transpose(np.array(ret, copy=True)).squeeze().fliplr()
        value = ret_copy[:, 0]
        for i in range(len(self.exercise_dates[:-1])):
            design = PolynomialFeatures(degree=4)
            linreg = LinearRegression(fit_intercept=False)
            index = ret_copy[:, i+1] > 0
            x = design.fit_transform(ret_copy[index, i+1])
            linreg.fit(x, value[index])
            estimates = linreg.predict(design.transform(ret_copy[:, i+1].reshape(-1, 1)))
            value = np.where(estimates > ret[:, i+1])



        ret = tf.squeeze(tf.transpose(ret))
        pv0 = tf.reshape(ret[:, 0], (-1, 1))
        pv1 = tf.reshape(ret[:, 1], (-1, 1))
        i = tf.where(tf.greater(ret[:, 0], 0))
        x = tf.gather(ret[:, 0], i)
        y = tf.gather(ret[:, 1], i)
        
        design = PolynomialFeatures(degree=4)
        x = design.fit_transform(x)
        lin_model = LinearRegression(fit_intercept=False)
        lin_model.fit(x, y)
        estimates = lin_model.predict(design.transform(tf.reshape(ret[:, 0], (-1, 1))))
        p = tf.where(tf.cast(tf.maximum(estimates, 0), dtype=tf.float32) >= pv0, pv1, pv0)
        #print(tf.cast(tf.maximum(estimates, 0), dtype=tf.float32) > pv0)
        return p
        '''

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
