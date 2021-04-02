import warnings
import os
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.sparse
from enum import Enum
import random
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import FeatureHasher

from aggregated_models.aggdataset import FeaturesSet
from aggregated_models.basicmodels import Hasher


class RPPrivateLogistic:
    def __init__(
        self,
        label,
        features,
        crossfeatures,
        train,
        lambdaL2=1.0,
        epsilon=1.0,
        projectionSize=50,
        hashspace=2 ** 16,
        max_iter=50,
        norm=None,
    ):
        import diffprivlib

        self.label = label
        self.featuresSet = FeaturesSet(features, crossfeatures, train)
        self.features = [x for x in self.featuresSet.mappings]
        self.lambdaL2 = lambdaL2
        self.epsilon = epsilon
        self.hasher = Hasher(self.features, hashspace)
        self.model = diffprivlib.models.LogisticRegression(epsilon=epsilon, C=0.5 / lambdaL2, data_norm=1.0)
        self.param = np.zeros(projectionSize + 1)
        self.delta = 0
        self.projectionSize = projectionSize
        self.G = np.random.normal(0, 1, [hashspace, self.projectionSize])

    def getHashedX(self, df):
        dfWithCfs = self.featuresSet.transformDf(df)
        X = self.hasher.hash_features(dfWithCfs)
        return X

    def getX(self, df):
        X = self.getHashedX(df) * self.G
        norms = np.linalg.norm(X, axis=1)
        return X / norms[:, np.newaxis]

    def getY(self, df):
        return df[self.label].values

    def fit(self, df, nbiter=50):
        x = self.getX(df)
        y = self.getY(df)
        with warnings.catch_warnings(record=True):
            self.model.fit(x, y)

    def predict_proba(self, df):
        x = self.getX(df)
        a = self.model.predict_proba(x)[:, 1]
        return a
        # return self.predict(x)

    def predict(self, x):
        logits = x * self.param[:-1] + self.param[-1]
        return 1.0 / (1.0 + np.exp(-logits))

    def initnoise(self, train):
        self._function_sensitivity = 0.25
        self._data_sensitivity = self.norm
        # self._data_sensitivity = 1.0
        self._alpha = self.lambdaL2 * 2  # *2 ?  / 2 ?
        epsilon_p = self.epsilon - 2 * np.log(
            1 + self._function_sensitivity * self._data_sensitivity / (0.5 * self._alpha)
        )
        delta = 0
        if epsilon_p <= 0:
            delta = (
                self._function_sensitivity * self._data_sensitivity / (np.exp(self.epsilon / 4) - 1) - 0.5 * self._alpha
            )
            epsilon_p = self.epsilon / 2
        scale = (epsilon_p / 2 / self._data_sensitivity) if self._data_sensitivity > 0 else float("inf")
        self.epsilon_p = epsilon_p
        _vector_dim = len(self.param) - 1
        normed_noisy_vector = np.random.normal(0, 1, _vector_dim)
        norm = np.linalg.norm(normed_noisy_vector, 2)
        noisy_norm = np.random.gamma(_vector_dim, 1 / scale, 1)
        normed_noisy_vector = normed_noisy_vector / norm * noisy_norm
        self.noisy_norm = noisy_norm
        self.normed_noisy_vector = normed_noisy_vector
        self.delta = delta
        print(f"scale:{scale} , noisy_norm:{noisy_norm} ,sigma:{noisy_norm/np.sqrt(_vector_dim)} ")

    def computeGradient(self, x, y):
        g = (2 * self.lambdaL2 + self.delta) * self.param
        p = self.predict(x)
        gllh = (p - y) * x
        if self.epsilon > 0:
            gllh += self.normed_noisy_vector

        g[:-1] += gllh
        g[-1] += sum(p - y)

        return g

    def computeLoss(self, x, y):
        regul = np.dot(self.param, self.param) * (self.lambdaL2 + 0.5 * self.delta)
        p = self.predict(x)
        llh = -(y * np.log(p + 1e-15) + (1 - y) * np.log(1 + 1e-15 - p)).sum()
        if self.epsilon > 0:
            llh += np.dot(self.normed_noisy_vector, self.param[:-1])
        return regul + llh

    def fit_alamain(self, df, nbiter=50):
        x = self.getX(df)
        y = self.getY(df)

        from scipy.optimize import minimize

        def myloss(w):
            self.param = w
            llh = self.computeLoss(x, y)
            # print( sum(w), llh )
            return llh

        def mygrad(w):
            self.param = w
            return self.computeGradient(x, y)

        optimresult = minimize(
            myloss,
            self.param,
            method="L-BFGS-B",
            jac=mygrad,
            options={"maxiter": nbiter},
        )
        self.param = optimresult.x
        return optimresult

    def predictDF(self, df):
        df = df.copy()
        df["pclick"] = self.predict_proba(df)
        return df

    def computeLossAndRegul(self, train):
        p = logisticCfs.predict_proba(train)
        y = train.click.values
        llh = -sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        p0 = sum(y) / len(train)
        llh -= -sum(y * np.log(p0) + (1 - y) * np.log(1 - p0))
        w = a = logisticCfs.model.model.coef_[0]
        regul = w.dot(w) * 200 / self.nbCoefs
        return np.array([llh + regul, llh, regul])
