import pandas as pd
import numpy as np
from aggregated_models.featuremappings import (
    CrossFeaturesMapping,
    SingleFeatureMapping,
)
from aggregated_models.SampleSet import SampleSet
from aggregated_models import featuremappings
from aggregated_models import Optimizers


class WeightsSet:
    """Map a feature to a subarray in an array of parameters. Usefull to build linear models.
    feature: a SingleFeatureMapping or CrossFeaturesMapping
    offset: first index used for this feature. subarray will be [offset:offset+feature.Size]
    """

    def __init__(self, feature: SingleFeatureMapping, offset):
        self.feature = feature
        self.offset = offset
        self.indices = np.arange(self.offset, self.offset + self.feature.Size)

    def GetIndices_(self, x: np.array):
        """x: np.array of shape (nb features, nb samples)
        return for each sample the index in param vector of (wrapped feature modality)"""
        return self.feature.Values_(x) + self.offset

    def GetIndices(self, df: pd.DataFrame):
        """df: dataframe of samples, containing one col for wrapped feature
        return for each line of df the index of (wrapped feature modality)
        """
        return self.feature.Values(df) + self.offset

    def __repr__(self):
        return f"WeightsSet on {self.feature} offset={self.offset}"


# Abstract class for models learned from agg data
class BaseAggModel:
    def __init__(self, aggdata, features):
        self.aggdata = aggdata
        self.features = features
        self.label = self.aggdata.label
        self.bestLoss = 9999999999999999.0
        self.nbEvals = 0
        self.normgrad = 0.0

    def transformDf(self, df):
        return self.aggdata.featuresSet.transformDf(df, False)

    def getMapping(self, var):
        return self.aggdata.featuresSet.getMapping(var)

    def prepareWeights(self, featuresAndCfs, offset=0):
        weights = {}
        for var in featuresAndCfs:
            featureMapping = self.getMapping(var)
            weights[var] = WeightsSet(featureMapping, offset)
            offset += featureMapping.Size
        return weights, offset

    def setparameters(self, x):
        if any(self.parameters != x):
            self.parameters = x
            self.update()

    def dotproducts(self, weights, x):
        results = np.zeros(x.shape[1])
        for w in weights.values():
            results += self.parameters[w.GetIndices_(x)]
        return results

    def dotproductsOnDF(self, weights, df):
        results = np.zeros(len(df))
        for var, w in weights.items():
            results += self.parameters[w.GetIndices(df)]
        return results

    def getAggDataVector(self, weights, projections):
        x = self.parameters * 0
        for var, w in weights.items():
            proj = projections[var]
            x[w.indices] = proj.Data
        return x

    def predictDF(self, df):
        df = self.transformDf(df)
        return self.predictDFinternal(df)

    def computeGradientAt(self, x):
        self.setparameters(x)
        return self.computeGradient()

    def update(self):
        raise NotImplementedError

    def predictDFinternal(self, df):
        raise NotImplementedError

    def computeGradient(self):
        raise NotImplementedError

    def computeLoss(self, epsilon=1e-12):
        return 0.0
