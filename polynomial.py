import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from models import Bachelier_helper, BlackScholes_helper, Bachelier_eulerScheme, BlackScholes_eulerScheme, simulate_data
from options import Call, Call_basket, Call_geometric

class PolyReg:
    def __init__(self, x, y, p):
        self.x = x
        self.y = y
        self.p = p
        self.poly_features = PolynomialFeatures(degree=p)
        self.X = self.poly_features.fit_transform(x)
        self.params = None

    def predict(self, x):
        X = self.poly_features.fit_transform(x)
        return np.dot(X, self.params)

    def differential(self, x):
        X = self.poly_features.fit_transform(x)
        powers = self.poly_features.powers_#.reshape(-1, 1)
        D = np.transpose(np.transpose(powers)[:, :, np.newaxis] * np.transpose(X), axes=(0, 2, 1)) / \
            np.transpose(x[:, :, np.newaxis], axes=(1, 0, 2))

        return np.sum(np.dot(D, self.params), axis=0).reshape(-1, 1)
        #return np.dot(D, self.params).reshape(-1, 1)

    def integrated(self, x):
        X = self.poly_features.fit_transform(x)
        powers = self.poly_features.powers_ + 1
        D = np.transpose(np.transpose(X) / np.transpose(powers)[:, :, np.newaxis], axes=(0, 2, 1)) * \
            np.transpose(x[:, :, np.newaxis], axes=(1, 0, 2))

        return np.dot(D, self.params).reshape(-1, 1)

    def fit(self, lmbda=0):
        n = np.shape(self.X)[1]
        X_t = np.transpose(self.X)
        self.params = np.dot(np.linalg.solve(np.dot(X_t, self.X) + np.identity(n) * lmbda, X_t), self.y)
        return self.params

    def fit_differential(self, Z, lmbda=None):
        if lmbda is None:
            lmbda = np.dot(np.transpose(y), y) / np.dot(np.transpose(Z[0, :, :]), Z[0, :, :])

        X_t = np.transpose(self.X)
        powers = self.poly_features.powers_
        D = np.transpose(np.transpose(powers)[:, :, np.newaxis] * np.transpose(self.X), axes=(0, 2, 1)) / \
            np.transpose(self.x[:, :, np.newaxis], axes=(1, 0, 2))
        D_t = np.transpose(D, axes=(0, 2, 1))
        Z = np.transpose(Z[:, :, np.newaxis], axes=(1, 0, 2))
        p1 = np.dot(X_t, self.X) + lmbda * np.sum(np.matmul(D_t, D), axis=0)
        p2 = np.dot(X_t, self.y) + lmbda * np.sum(np.matmul(D_t, Z), axis=0)
        self.params = np.linalg.solve(p1, p2)
        return self.params

