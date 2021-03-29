import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import FeatureHasher
from aggregated_models.featuremappings import FeaturesSet

# Wrapper around sklearn.feature_extraction.FeatureHasher
class Hasher:
    def __init__(self, features, hashspace=2 ** 16):
        self.features = features
        self.hashspace = hashspace

    def __str__(self):
        return "features = " + str(self.features) + "; hashspace =" + str(self.hashspace)

    def features_to_list_of_strings(self, row):
        return [f"{feature}_{row[feature]}" for feature in row.index]

    def hash_features(self, df):
        raw_features = df[self.features]
        features_as_list_of_strings = raw_features.apply(self.features_to_list_of_strings, axis=1)
        hasher = FeatureHasher(n_features=self.hashspace, input_type="string", alternate_sign=False)
        features = hasher.fit_transform(features_as_list_of_strings)
        return features


# simple logistic regression from hashed data. No crossfeatures.
# wrapper around sklearn.linear_model.LogisticRegression
class LogisticModel:
    def __init__(
        self,
        label,
        features,
        lambdaL2=1.0,
        hashspace=2 ** 16,
        max_iter=50,
        epsilon=None,
    ):
        if epsilon is None:
            self.model = LogisticRegression(
                max_iter=max_iter, C=0.5 / lambdaL2
            )  # multiplying by 0.5 to get results similar to my own reimplem ??
        else:
            self.model = diffprivlib.models.LogisticRegression(epsilon=epsilon, max_iter=max_iter, C=0.5 / lambdaL2)
        self.label = label
        self.features = features
        self.lambdaL2 = lambdaL2
        self.hasher = Hasher(features, hashspace)
        self.epsilon = epsilon

    def fit(self, df):
        labels = df[self.label]
        featuresdf = self.hasher.hash_features(df)
        with warnings.catch_warnings(record=True):
            self.model.fit(featuresdf, labels)

    def predict_proba(self, df):
        featuresdf = self.hasher.hash_features(df)
        return self.model.predict_proba(featuresdf)[:, 1]

    def predictDF(self, df):
        df = df.copy()
        df["pclick"] = self.predict_proba(df)
        return df


# Logistic regression with cross features (ie 2e order kernell)
class LogisticModelWithCF:
    def __init__(
        self,
        label,
        features,
        crossfeatures,
        train,
        lambdaL2=1.0,
        hashspace=2 ** 16,
        max_iter=50,
        epsilon=None,
    ):
        self.featuresSet = FeaturesSet(features, crossfeatures, train)
        self.features = [x for x in self.featuresSet.mappings]
        self.model = LogisticModel(label, self.features, lambdaL2, hashspace, max_iter, epsilon=epsilon)
        self.lambdaL2 = lambdaL2

    def fit(self, df):
        dfWithCfs = self.featuresSet.transformDf(df, True)
        self.model.fit(dfWithCfs)

    def predict_proba(self, df):
        dfWithCfs = self.featuresSet.transformDf(df)
        return self.model.predict_proba(dfWithCfs)

    def predictDF(self, df):
        dfWithCfs = self.featuresSet.transformDf(df, True)
        return self.model.predictDF(dfWithCfs)

    def computeLossAndRegul(self, train):
        p = logisticCfs.predict_proba(train)
        y = train.click.values
        llh = -sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        p0 = sum(y) / len(train)
        llh -= -sum(y * np.log(p0) + (1 - y) * np.log(1 - p0))
        w = a = logisticCfs.model.model.coef_[0]
        regul = w.dot(w) * 200 / self.nbCoefs
        return np.array([llh + regul, llh, regul])


class NaiveBayesModel:
    def __init__(self, label, features, lambdaL2=1.0):
        self.label = label
        self.features = features
        self.lambdaL2 = lambdaL2

    def fit(self, df):
        labels = df[self.label]
        self.models = {}
        for var in self.features:
            self.models[var] = LogisticModel(self.label, features=[var], lambdaL2=self.lambdaL2)
            self.models[var].fit(df)

    def predict_proba(self, df):
        y = df[self.label].values
        py = y.sum() / len(df)
        unnormalizedP0 = y * 0 + 1 - py
        unnormalizedP1 = y * 0 + py
        for var in self.features:
            predictions = self.models[var].predict_proba(df)
            unnormalizedP1 = unnormalizedP1 * predictions / py
            unnormalizedP0 = unnormalizedP0 * (1 - predictions) / (1 - py)

        return unnormalizedP1 * 1.0 / (unnormalizedP1 + unnormalizedP0)

    def predictDF(self, df):
        df = df.copy()
        df["pclick"] = self.predict_proba(df)
        return df

    def computeLLH(self, df):
        y = df[self.label].values
        predictions = self.predict_proba(df)
        return LlhCVN(predictions, y)
