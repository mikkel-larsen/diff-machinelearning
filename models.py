import numpy as np
from numpy import random
from scipy.stats import norm
from scipy.stats import truncnorm
from scipy.stats import multivariate_normal
from scipy.stats.qmc import MultivariateNormalQMC
import tensorflow as tf
from scipy.optimize import root_scalar
from scipy.integrate import quad

class Multivariate_0mean_Normal:
    def __init__(self, dim=1, var=None, antithetic=False, quasi_mc=False):
        self.dim = dim
        if var is None:
            self.var = np.identity(dim)
        else:
            self.var = var
        self.antithetic = antithetic
        self.quasi_mc = quasi_mc

        if quasi_mc:
            self.gen = MultivariateNormalQMC(np.zeros(self.dim), self.var)
            self.shuffler = np.random.default_rng()
        elif antithetic:
            self.gen = multivariate_normal(np.zeros(self.dim), self.var)
        else:
            self.gen = multivariate_normal(np.zeros(self.dim), self.var)

    def generate_random(self, n=1):
        if self.quasi_mc:
            z = self.gen.random(n).reshape((n, self.dim))
            perm = np.arange(n)
            self.shuffler.shuffle(perm)
            return z[perm]
        elif self.antithetic:
            n_half = (n + 1) // 2
            z = self.gen.rvs(size=n_half).reshape(n, self.dim)
            return np.row_stack((z, -z))[:n]
        else:
            return self.gen.rvs(size=n).reshape(n, self.dim)


class EulerScheme:  # Abstract base class for different models
    def initialize(self, **kwargs):
        # Initialize stuff that do not need to be
        # calculated in every iteration of euler scheme
        pass

    def update_price_path(self, **kwargs):
        raise NotImplementedError  # A single euler step


class Bachelier_eulerScheme(EulerScheme):
    def __init__(self, rf_rate, var, antithetic=False, quasi_mc=False):
        self.const = None
        self.dt = None
        self.dt_sqrt = None
        self.gen = None
        self.n = None
        self.m = None
        self.antithetic = antithetic
        self.quasi_mc = quasi_mc
        self.rf_rate = rf_rate
        self.var = var

    def initialize(self, **kwargs):
        self.dt = kwargs['dt']
        self.dt_sqrt = np.sqrt(self.dt)
        self.const = 1 + self.rf_rate * self.dt
        self.n, self.m = np.shape(kwargs['s0'])
        self.gen = Multivariate_0mean_Normal(self.m, self.var, self.antithetic, self.quasi_mc)

    def update_price_path(self, s0, **kwargs):
        z = self.gen.generate_random(self.n)
        return s0 * self.const + self.dt_sqrt * z


class BlackScholes_eulerScheme(EulerScheme):
    def __init__(self, rf_rate, var, antithetic=False, quasi_mc=False):
        self.const = None
        self.dt = None
        self.dt_sqrt = None
        self.gen = None
        self.n = None
        self.m = None
        self.rf_rate = rf_rate
        self.var = var
        self.var_diag = None
        self.antithetic = antithetic
        self.quasi_mc = quasi_mc

    def initialize(self, **kwargs):
        self.dt = kwargs['dt']
        self.dt_sqrt = np.sqrt(self.dt)
        self.const = 1 + self.rf_rate * self.dt
        self.n, self.m = np.shape(kwargs['s0'])
        self.gen = Multivariate_0mean_Normal(self.m, self.var, self.antithetic, self.quasi_mc)

    def simulate_endpoint(self, s0, T):
        shape = np.shape(s0)
        n, m = shape

        gen = Multivariate_0mean_Normal(m, self.var, self.antithetic, self.quasi_mc)
        z = gen.generate_random(n)

        if type(self.var) is not list and type(self.var) is not np.ndarray:
            var_diag = self.var
        else:
            var_diag = np.diag(self.var)

        return s0 * np.exp((self.rf_rate - var_diag / 2.0) * T + np.sqrt(T) * z)

    def update_price_path(self, s0, **kwargs):
        z = self.gen.generate_random(self.n)
        return s0 * (self.const + self.dt_sqrt * z)


class Vasicek_eulerScheme(EulerScheme):
    def __init__(self, kappa, theta, var, antithetic=False, quasi_mc=False):
        self.const = None
        self.dt = None
        self.dt_sqrt = None
        self.gen = None
        self.n = None
        self.m = None
        self.kappa = kappa
        self.theta = theta
        self.var = var
        self.antithetic = antithetic
        self.quasi_mc = quasi_mc

    def initialize(self, **kwargs):
        self.dt = kwargs['dt']
        self.dt_sqrt = np.sqrt(self.dt)
        self.n, self.m = np.shape(kwargs['s0'])
        self.gen = Multivariate_0mean_Normal(self.m, self.var, self.antithetic, self.quasi_mc)

    def update_price_path(self, s0, **kwargs):
        z = self.gen.generate_random(self.n)
        return s0 + self.kappa * (self.theta - s0) * self.dt + self.dt_sqrt * z


class G2pp_eulerScheme(EulerScheme):
    def __init__(self, a1, a2, sigma1, sigma2, rho, antithetic=False, quasi_mc=False):
        self.const = None
        self.dt = None
        self.dt_sqrt = None
        self.gen = None
        self.n = None
        self.m = None
        self.a1 = a1
        self.a2 = a2
        self.a = np.array([a1, a2])
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma = np.array([sigma1, sigma2])
        self.rho = rho
        cov = rho * sigma1 * sigma2
        self.var = np.array([[sigma1**2, cov], [cov, sigma2**2]])
        self.antithetic = antithetic
        self.quasi_mc = quasi_mc

    def initialize(self, **kwargs):
        self.dt = kwargs['dt']
        self.dt_sqrt = np.sqrt(self.dt)
        self.n, self.m = np.shape(kwargs['s0'])
        self.gen = Multivariate_0mean_Normal(self.m, self.var, self.antithetic, self.quasi_mc)

    def update_price_path(self, s0, **kwargs):
        z = self.gen.generate_random(self.n)
        return s0 * (1 - self.a * self.dt) + self.dt_sqrt * z


def simulate_data(n, rng, option, model, n_updates=None, seed=None):
    if seed is not None:  # Possibility of setting seed for reproducibility
        tf.random.set_seed(seed)
        np.random.seed(seed)

    if type(n) is not list and type(n) is not np.ndarray:
        n = [n, 1]  # Failsafe: random number generators takes lists as input

    if np.size(rng) == 2:  # Randomly generate spots in given range
        spot_min, spot_max = rng
    else:
        spot_min = rng  # If only kappa single number is given all spots will start here
        spot_max = rng

    is_simple = False  # Check if derivative is simple (and vocal about it)
    if hasattr(option, 'is_simple'):
        if option.is_simple:
            is_simple = True

    T = option.T
    s0 = tf.random.uniform(n, spot_min, spot_max, dtype=tf.float32)  # Randomly select spots

    if is_simple and hasattr(model, 'simulate_endpoint'):  # Skip euler steps if possible
        with tf.GradientTape() as tape:
            #s0 = tf.random.uniform(n, spot_min, spot_max, dtype=tf.float32)  # Randomly select spots
            tape.watch(s0)  # Spots are not tf.Variables, we tell TF to track anyway
            sT = model.simulate_endpoint(s0, T)
            payoff = option.payoff(sT)
        Z = tape.gradient(payoff, s0)
    else:
        if n_updates is None:
            end = int(T * 252)  # Update once every day
            dt = 1 / 252
        else:
            end = n_updates
            dt = T / end

        if hasattr(option, 'initialize'):
            option.initialize(dt=dt, end=end, T=T, n=n)

        #s0 = tf.random.uniform(n, spot_min, spot_max, dtype=tf.float32)  # Randomly select spots
        # Initialize constants that do not need to be recalculated for every ..
        model.initialize(dt=dt, s0=s0, T=T, end=end, n=n)  # .. iteration in euler scheme update
        with tf.GradientTape() as tape:
            tape.watch(s0)  # Spots are not tf.Variables, we tell TF to track anyway
            st = s0
            price_path = [st]  # List to hold price-path
            for _ in range(end):
                st = model.update_price_path(st, dt=dt)  # Update with single euler step
                price_path.append(st)  # Add to price-path

            if is_simple:  # Find payoff
                payoff = option.payoff(price_path[-1], dt=dt, end=end, n=n)
            else:
                payoff = option.payoff(price_path, dt=dt, end=end, n=n)

        # Get gradient of payoff w.r.t. spot (s0)
        Z = tape.gradient(payoff, s0)
        payoff = tf.reshape(payoff, (n[0], -1))

    return np.array(s0, copy=False), np.array(payoff, copy=False), np.array(Z, copy=False)


###########################################################################################
#################################### HELPER FUNCTIONS #####################################
###########################################################################################

class G2pp_helper:
    def __init__(self, a1, a2, sigma1, sigma2, rho):
        self.a1 = a1
        self.a2 = a2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.rho = rho

    def A(self, t, T):
        timetomat = T - t
        B1 = (1 - np.exp(-self.a1 * timetomat)) / self.a1
        B2 = (1 - np.exp(-self.a2 * timetomat)) / self.a2
        B12 = (1 - np.exp(-(self.a1 + self.a2) * timetomat)) / (self.a1 + self.a2)
        V_sq = (self.sigma1 / self.a1)**2 * (timetomat - B1 - self.a1 * 0.5 * B1**2) + \
               (self.sigma2 / self.a2)**2 * (timetomat - B2 - self.a2 * 0.5 * B2**2) + \
               2 * self.sigma1 * self.sigma2 * self.rho / (self.a1 * self.a2) * (timetomat - B1 - B2 + B12)
        return np.exp(0.5 * V_sq)

    def B1(self, t, T):
        timetomat = T - t
        return (1 - np.exp(-self.a1 * timetomat)) / self.a1

    def B2(self, t, T):
        timetomat = T - t
        return (1 - np.exp(-self.a2 * timetomat)) / self.a2

    def B12(self, t, T):
        timetomat = T - t
        return (1 - np.exp(-(self.a1 + self.a2) * timetomat)) / (self.a1 + self.a2)

    def p(self, t, T, r):
        r1, r2 = np.transpose(r)
        timetomat = T - t
        B1 = self.B1(t, T)
        B2 = self.B2(t, T)
        B12 = self.B12(t, T)
        V_sq = (self.sigma1 / self.a1)**2 * (timetomat - B1 - self.a1 * 0.5 * B1**2) + \
               (self.sigma2 / self.a2)**2 * (timetomat - B2 - self.a2 * 0.5 * B2**2) + \
               2 * self.sigma1 * self.sigma2 * self.rho / (self.a1 * self.a2) * (timetomat - B1 - B2 + B12)
        M = r1 * B1 + r2 * B2
        return np.exp(-M + 0.5 * V_sq)

    def swaption_price(self, T, K, settlement_dates, r, opt_type='payer'):
        if opt_type == 'receiver':
            w = -1
        else:
            w = 1
        B1 = self.B1(0, T)
        B2 = self.B2(0, T)
        B12 = self.B12(0, T)
        sigma_tilde_1 = self.sigma1 * np.sqrt((1 - np.exp(-2 * self.a1 * T)) / (2 * self.a1))
        sigma_tilde_2 = self.sigma2 * np.sqrt((1 - np.exp(-2 * self.a2 * T)) / (2 * self.a2))
        rho_tilde = self.sigma1 * self.sigma2 * self.rho / (sigma_tilde_1 * sigma_tilde_2) * B12
        mu_tilde_1 = self.sigma1**2 / (2 * self.a1**2) * (1 - np.exp(-2 * self.a1 * T)) + \
                     self.sigma1 * self.sigma2 * self.rho / self.a2 * B12 - \
                     (self.sigma1**2 / self.a1 + self.sigma1 * self.sigma2 * self.rho / self.a2) * B1
        mu_tilde_2 = self.sigma2**2 / (2 * self.a2 ** 2) * (1 - np.exp(-2 * self.a2 * T)) + \
                     self.sigma1 * self.sigma2 * self.rho / self.a1 * B12 - \
                     (self.sigma2**2 / self.a2 + self.sigma1 * self.sigma2 * self.rho / self.a1) * B2

        def kappav(x):
            return - self.B2(T, settlement_dates) * \
                   (mu_tilde_2 - sigma_tilde_2**2 * (1 - rho_tilde**2) * 0.5 *
                    self.B2(T, settlement_dates) + rho_tilde * sigma_tilde_2 *
                    (x - mu_tilde_1) / sigma_tilde_1)

        cv = K * (settlement_dates - np.concatenate((T, settlement_dates[:-1])))
        cv[-1] += 1

        def lambdav(x):
            return cv * self.A(T, settlement_dates) * np.exp(-x * self.B1(T, settlement_dates))

        def x_bar(xx):
            def f(r):
                return np.sum(lambdav(xx) * np.exp(-r * self.B2(T, settlement_dates))) - 1
            return root_scalar(f, bracket=(-2, 2)).root

        def h1(x):
            return (x_bar(x) - mu_tilde_2) / (sigma_tilde_2 * np.sqrt(1 - rho_tilde**2)) - \
                   rho_tilde * (x - mu_tilde_1) / (sigma_tilde_1 * np.sqrt(1 - rho_tilde**2))

        def h2v(x):
            return h1(x) + self.B2(T, settlement_dates) * sigma_tilde_2 * np.sqrt(1 - rho_tilde**2)

        grid = np.linspace(mu_tilde_1 - 0 * sigma_tilde_1, mu_tilde_1 + 6 * sigma_tilde_1, 200)
        dx = grid[1] - grid[0]

        def func(x, r):
            return self.p(0, T, r) * np.exp(-0.5 * ((x - mu_tilde_1) / sigma_tilde_1)**2) / (sigma_tilde_1 * np.sqrt(2 * 3.141592)) * \
                   (norm.cdf(-w * h1(x)) - np.sum(lambdav(x) * np.exp(kappav(x)) * norm.cdf(-w * h2v(x))))

        ret = []
        for rate in r:
            tmp = 0
            for x in grid:
                tmp += np.exp(-0.5 * ((x - mu_tilde_1) / sigma_tilde_1)**2) * \
                       (norm.cdf(-w * h1(x)) - np.sum(lambdav(x) * np.exp(kappav(x)) * norm.cdf(-w * h2v(x))))

            #print(quad(func, mu_tilde_1, mu_tilde_1 + 6 * sigma_tilde_1, args=(rate)))
            ret.append(w * tmp * dx * self.p(0, T, rate) / (sigma_tilde_1 * np.sqrt(2 * 3.141592)))

        return np.squeeze(ret)


class Vasicek_helper:
    def __init__(self, vol, kappa, theta):
        self.vol = vol
        self.kappa = kappa
        self.theta = theta

    def p(self, t, T, r):
        B = (1 - np.exp(-self.kappa * (T - t))) / self.kappa
        A = (B - T + t) * (self.theta - 0.5 * self.vol ** 2 / self.kappa ** 2) - (self.vol * B) ** 2 / (4 * self.kappa)
        return np.exp(A - B * r)

    def r_star(self, c, t, T, equal_to=1):
        def f(r, t, T):
            return np.sum(c * self.p(t, T, r)) - equal_to
        return root_scalar(f, args=(t, T), bracket=(-2, 2)).root

    def bond_option(self, t, T, K, S, r, opt_type='call'):
        ptT = self.p(t, T, r)
        ptS = self.p(t, S, r)
        sigma_p = (1 - np.exp(-self.kappa * (S - T))) / self.kappa * np.sqrt(self.vol ** 2 / (2 * self.kappa) * (1 - np.exp(-2 * self.kappa * (T - t))))
        d = np.log(ptS / (ptT * K)) / sigma_p + sigma_p / 2

        if opt_type == 'put':
            return ptT * K * norm.cdf(-d + sigma_p) - ptS * norm.cdf(-d)
        return ptS * norm.cdf(d) - ptT * K * norm.cdf(d - sigma_p)

    def swaption_price(self, T, K, settlement_dates, r, opt_type='receiver'):
        n = len(settlement_dates)
        coverage = settlement_dates[1] - settlement_dates[0]
        coupons = np.ones(n) * coverage * K
        coupons[-1] += 1
        r_star = self.r_star(coupons, T, settlement_dates)
        strikes = self.p(T, settlement_dates, r_star)

        if opt_type == 'payer':
            if type(r) is int or type(r) is float:
                return np.sum(coupons * self.bond_option(0, T, strikes, settlement_dates, r, 'put'))
            else:
                return np.array([np.sum(coupons * self.bond_option(0, T, strikes, settlement_dates, i, 'put')) for i in r])

        if type(r) is int or type(r) is float:
            return np.sum(coupons * self.bond_option(0, T, strikes, settlement_dates, r))
        else:
            return np.array([np.sum(coupons * self.bond_option(0, T, strikes, settlement_dates, i)) for i in r])


class Bachelier_helper:
    def __init__(self, rf_rate=0.0, vol=None):
        self.rf_rate = rf_rate
        self.vol = vol

    def call_price(self, spot, strike, T, vol=None, rf_rate=None):
        if vol is None:
            vol = self.vol
        if rf_rate is None:
            rf_rate = self.rf_rate

        if rf_rate == 0:
            diff = spot - strike
            const = vol * np.sqrt(T)
            return diff * norm.cdf(diff / const) + const * norm.pdf(diff / const)

        D = (spot * np.exp(rf_rate * T) - strike) / (vol * np.sqrt(T))
        return np.exp(-rf_rate * T) * vol * np.sqrt(T) * (D * norm.cdf(D) + norm.pdf(D))

    def call_delta(self, spot, strike, T, vol=None, rf_rate=None):
        if vol is None:
            vol = self.vol
        if rf_rate is None:
            rf_rate = self.rf_rate

        d = (spot - strike) / vol / np.sqrt(T)
        return norm.cdf(d)

    def simulate_price_path(self, n, spot, T, n_yearly_updates=252):
        if np.size(spot) == 2:
            spot_min, spot_max = spot
        else:
            spot_min = spot
            spot_max = spot

        if T < 1:
            end = n_yearly_updates  # Update 252 times (default)
            dt = T / n_yearly_updates
        else:
            end = int(T * n_yearly_updates)  # Update once every day (default)
            dt = 1 / n_yearly_updates

        spots = np.zeros((n, end))
        spots[:, 0] = random.uniform(spot_min, spot_max, n)

        z = random.standard_normal((n, end))
        dt_sqrt = np.sqrt(dt)
        for i in range(1, end):
            spots[:, i] = spots[:, i - 1] * (1 + self.rf_rate * dt) + self.vol * dt_sqrt * z[:, i]

        return spots

    def simulate_basket(self, n, m, rng, option, w, cov, seed=None):
        if seed is not None:
            tf.random.set_seed(seed)
        if np.size(rng) == 2:
            spot_min, spot_max = rng
        else:
            spot_min = rng
            spot_max = rng

        with tf.GradientTape() as tape:
            T = option.T
            shape = [n, m]
            s0 = tf.random.uniform(shape, spot_min, spot_max, dtype=tf.float64)
            tape.watch(s0)
            b0 = s0 @ w.reshape(-1, 1)
            order = tf.reshape(tf.argsort(b0, axis=0), -1)
            s0 = tf.gather(s0, order)
            b0 = tf.sort(b0, axis=0)

            sT = s0
            mean = tf.zeros(m)
            if self.rf_rate == 0.0:
                sT += np.sqrt(T) * random.multivariate_normal(mean, cov, n)
            else:
                if T < 1:
                    end = tf.constant(252, dtype=tf.int64)  # Update 252 times
                    dt = tf.constant(T / 252, dtype=tf.float64)
                else:
                    end = tf.constant(T * 252, dtype=tf.int64)  # Update once every day times
                    dt = tf.constant(1 / 252, dtype=tf.float64)

                dt_sq = tf.sqrt(dt)
                for _ in tf.range(end):
                    z = random.multivariate_normal(mean, cov, n)
                    sT += sT * self.rf_rate * dt + dt_sq * z

            bT = sT @ w.reshape(-1, 1)

            payoff = option.payoff(bT)
            Z = tape.gradient(payoff, s0)

        return np.array(s0), np.array(b0), np.array(payoff), np.array(Z)


class BlackScholes_helper:
    def __init__(self, rf_rate, vol):
        self.rf_rate = rf_rate
        self.vol = vol

    def call_price(self, spot, strike, T, vol=None, rf_rate=None):
        if vol is None:
            vol = self.vol
        if rf_rate is None:
            rf_rate = self.rf_rate

        d1 = (np.log(spot / strike) + (rf_rate + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        return spot * norm.cdf(d1) - strike * np.exp(-rf_rate * T) * norm.cdf(d2)

    def call_delta(self, spot, strike, T):
        d1 = (np.log(spot / strike) + (self.rf_rate + 0.5 * self.vol ** 2) * T) / (self.vol * np.sqrt(T))
        return norm.cdf(d1)

    def put_price(self, spot, strike, T):
        d1 = (np.log(spot / strike) + (self.rf_rate + 0.5 * self.vol ** 2) * T) / (self.vol * np.sqrt(T))
        d2 = d1 - self.vol * np.sqrt(T)
        return strike * np.exp(-self.rf_rate * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)

    def put_delta(self, spot, strike, T):
        d1 = (np.log(spot / strike) + (self.rf_rate + 0.5 * self.vol ** 2) * T) / (self.vol * np.sqrt(T))
        return norm.cdf(d1) - 1

    def straddle_price(self, spot, strike, T):
        return self.call_price(spot, strike, T) + self.put_price(spot, strike, T)

    def straddle_delta(self, spot, strike, T):
        return self.call_delta(spot, strike, T) + self.put_delta(spot, strike, T)

    def digital_price(self, spot, strike, T):
        d1 = (np.log(spot / strike) + (self.rf_rate + 0.5 * self.vol ** 2) * T) / (self.vol * np.sqrt(T))
        d2 = d1 - self.vol * np.sqrt(T)
        return np.exp(-self.rf_rate * T) * norm.cdf(d2)

    def digital_delta(self, spot, strike, T):
        return np.exp(-self.rf_rate * T) * (norm.pdf((np.log(spot / strike) +
               (self.rf_rate + 0.5 * self.vol**2) * T) / (self.vol * np.sqrt(T)) -
               self.vol * np.sqrt(T)) * (1 / strike / (spot / strike) / (self.vol * np.sqrt(T))))


    def simulate_price_path(self, n, spot, T, n_yearly_updates=252, seed=None):
        random.seed(seed)
        if np.size(spot) == 2:
            spot_min, spot_max = spot
        else:
            spot_min = spot
            spot_max = spot

        if T < 1:
            end = n_yearly_updates  # Update 252 times (default)
            dt = T / n_yearly_updates
        else:
            end = int(T * n_yearly_updates)  # Update once every day (default)
            dt = 1 / n_yearly_updates

        spots = np.zeros((n, end))
        spots[:, 0] = random.uniform(spot_min, spot_max, n)
        z = random.standard_normal((n, end))
        dt_sqrt = np.sqrt(dt)
        for i in range(1, end):
            spots[:, i] = spots[:, i - 1] * (1 + self.rf_rate * dt + self.vol * dt_sqrt * z[:, i])

        return spots

    def simulate_data(self, n, rng, option, seed=None):
        if seed is not None:
            tf.random.set_seed(seed)

        T = option.T

        if np.size(rng) == 2:
            spot_min, spot_max = rng
        else:
            spot_min = rng
            spot_max = rng

        simple = False
        if hasattr(option, 'is_simple'):
            if option.is_simple:
                simple = True

        if simple:
            with tf.GradientTape() as tape:
                s0 = tf.sort(tf.random.uniform([n, 1], spot_min, spot_max, dtype=tf.float64), axis=0)
                tape.watch(s0)
                z = tf.random.normal([n, 1], dtype=tf.float64)
                sT = s0 * tf.exp((self.rf_rate - self.vol ** 2 / 2.0) * tf.cast(T, tf.float64) +
                                 self.vol * tf.sqrt(tf.cast(T, tf.float64)) * z)
                payoff = option.payoff(sT)
            Z = tape.gradient(payoff, s0)
        else:
            if T < 1:
                end = 252  # Update 252 times (default)
                dt = T / 252
            else:
                end = int(T * 252)  # Update once every day (default)
                dt = 1 / 252

            with tf.GradientTape() as tape:
                s0 = tf.sort(tf.random.uniform([1, n], spot_min, spot_max, dtype=tf.float64), axis=0)
                tape.watch(s0)
                z = random.standard_normal((n, end))
                dt_sqrt = np.sqrt(dt)
                sT = s0
                price_path = [sT]
                for i in range(end):
                    sT += sT * (self.rf_rate * dt + self.vol * dt_sqrt * z[:, i])
                    price_path.append(sT)

                payoff = option.payoff(price_path)
            Z = tape.gradient(payoff, s0)
            s0 = tf.reshape(s0, (-1, 1))
            payoff = tf.reshape(payoff, (-1, 1))
            Z = tf.reshape(Z, (-1, 1))

        return np.array(s0), np.array(payoff), np.array(Z)

    @staticmethod
    def simulate(n, min, max, option, vol, rf_rate, seed=None):
        if seed is not None:
            tf.random.set_seed(seed)

        T = option.T

        with tf.GradientTape() as tape:
            s0 = tf.sort(tf.random.uniform([n, 1], min, max, dtype=tf.float64), axis=0)
            tape.watch(s0)
            z = tf.random.normal([n, 1], dtype=tf.float64)
            sT = s0 * tf.exp(
                (rf_rate - vol ** 2 / 2.0) * tf.cast(T, tf.float64) + vol * tf.sqrt(tf.cast(T, tf.float64)) * z)
            payoff = option.payoff(sT)

        Z = tape.gradient(payoff, s0)

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
