import pandas as pd
import numpy as np
from agg_models.aggLogistic import AggLogistic

# model averaging several submodels
class EnsembleModel:
    def __init__(self, models, averaging):
        self.models = models
        if averaging == "logit":
            self.average = self.logitAverage
        else:
            self.average = self.arithmeticAverage

    def predictDF(self, df):
        predictions = [model.predictDF(df.copy()).pclick.values for model in self.models]
        df["pclick"] = self.average(predictions)
        return df

    def arithmeticAverage(self, predictions):
        return sum(predictions) / len(predictions)

    def logitAverage(self, predictions):
        logits = [np.log(x / (1 - x)) for x in predictions]
        logits = self.arithmeticAverage(logits)
        return 1 / (1 + np.exp(-logits))


# learing one logistic for each pair of features
class LogisticEnsemble(EnsembleModel):
    def __init__(self, aggdata, features, regulL2=1.0, averaging="logit"):
        self.features = features
        models = [
            AggLogistic(aggdata, [f1, f2], clicksCfs="*&*", regulL2=regulL2)
            for f1 in features
            for f2 in features
            if f1 < f2
        ]
        super().__init__(models, averaging)

    def fit(self, train):
        for model in self.models:
            model.fit(train)
            model.samples = []
