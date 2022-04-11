import numpy as np


class Preprocessor:
    def __init__(self, *args, **kwargs):
        self.y_sd = None
        self.y_mean = None
        self.x_sd = None
        self.x_mean = None

    def fit(self, x, y, z=None):
        x_mean, x_sd = np.mean(x, axis=0), np.std(x, axis=0)
        y_mean, y_sd = np.mean(y, axis=0), np.std(y, axis=0)
        self.x_mean, self.x_sd = x_mean, np.where(x_sd == 0.0, 1, x_sd)
        self.y_mean, self.y_sd = y_mean, np.where(y_sd == 0.0, 1, y_sd)

        return self

    def transform(self, x, y=None, z=None):
        x = (x - self.x_mean) / self.x_sd
        if y is None and z is None:
            return x

        if y is not None:
            y = (y - self.y_mean) / self.y_sd

        if z is not None:
            z = z * self.x_sd / self.y_sd
            return x, y, z
        return x, y

    def fit_transform(self, x, y, z=None):
        self.fit(x, y, z)
        return self.transform(x, y, z)

    def inverse_transform(self, x, y=None, z=None):
        x = x * self.x_sd + self.x_mean

        if y is not None:
            y = y * self.y_sd + self.y_mean

        if z is not None:
            z = z * self.y_sd / self.x_sd
            return x, y, z

        return x, y


class DPCA:
    def __init__(self, n_dim=None, *args, **kwargs):
        self.basis = None
        self.explained_variance = None
        self.eigen_values = None
        self.n_dim = n_dim

    def fit(self, z):
        if self.n_dim is None:
            self.n_dim = min(np.shape(z)[0], np.shape(z)[1])

        z = np.dot(z.T, z) / len(z)
        _, D, Pt = np.linalg.svd(z, full_matrices=False)

        self.eigen_values = D
        self.explained_variance = D / np.sum(D)
        cumsum_D = np.cumsum(D)
        if self.n_dim < 1:
            self.n_dim = np.searchsorted(cumsum_D, self.n_dim)
        self.basis = Pt.T[:, :self.n_dim]
        return self

    def transform(self, x, z=None):
        if z is None:
            return x @ self.basis
        return x @ self.basis, z @ self.basis

    def fit_transform(self, x, z):
        self.fit(z)
        return self.transform(x, z)

    def inverse_transform(self, x, z=None):
        if z is None:
            return x @ self.basis.T
        else:
            return x @ self.basis.T, z @ self.basis.T


class PCA:
    def __init__(self, n_dim=None, center=True):
        self.n_dim = n_dim
        self.center = center
        self.basis = None
        self.D = None
        self.X_mean = None
        self.eigen_values = None
        self.explained_variance = None

    def fit(self, X):
        if self.n_dim is None:
            self.n_dim = min(np.shape(X)[0], np.shape(X)[1])

        if self.center:
            self.X_mean = X.mean(axis=0)
            X = X - self.X_mean

        X = np.dot(X.T, X) / len(X)
        _, D, Pt = np.linalg.svd(X, full_matrices=False)

        self.eigen_values = D
        self.explained_variance = D / np.sum(D)
        cumsum_D = np.cumsum(D)
        if self.n_dim < 1:
            self.n_dim = np.searchsorted(cumsum_D, self.n_dim)
        self.basis = Pt.T[:, :self.n_dim]
        return self

    def transform(self, X, Z=None):
        if self.center:
            X = X - self.X_mean
        if Z is None:
            return X @ self.basis
        else:
            return X @ self.basis, Z @ self.basis

    def fit_transform(self, Z, X=None):
        self.fit(Z)
        if X is None:
            return self.transform(Z)
        else:
            return self.transform(Z, X)

    def inverse_transform(self, X, Z=None):
        if Z is None:
            return X @ self.basis.T
        else:
            return X @ self.basis.T, Z @ self.basis.T

    def reconstruct(self, X):
        return self.inverse_transform(self.transform(X))