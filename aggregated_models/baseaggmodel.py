import pandas as pd
import numpy as np

from aggregated_models.SampleSet import SampleSet
from aggregated_models.FeatureEncodings import *
from aggregated_models import Optimizers


class WeightsSet:
    """Map a feature to a subarray in an array of parameters. Usefull to build linear models.
    feature: a SingleFeatureMapping or CrossFeaturesMapping
    offset: first index used for this feature. subarray will be [offset:offset+feature.Size]
    """

    def __init__(self, feature: IEncoding, offset: int):
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

        aggdata.featuresSet.fix_fids(features)  # to work with a sublist of the features in aggdata

        self.aggdata = aggdata
        self.features = features
        self.label = self.aggdata.label
        self.bestLoss = 9999999999999999.0
        self.nbEvals = 0
        self.normgrad = 0.0
        self.parameters = None

    def transformDf(self, df):
        return self.aggdata.featuresSet.transformDf(df, False)

    def DfToX(self, df):
        df = self.transformDf(df)
        x = np.zeros((len(self.features), len(df)), dtype=np.int32)
        for f in self.features:
            encoding = self.aggdata.featuresSet.encodings[f]
            x[encoding._fid] = encoding.Values(df)
        return x

    def getEncoding(self, var):
        return self.aggdata.aggDisplays[var].feature

    def prepareWeights(self, featuresAndCfs, offset=0):
        weights = {}
        for var in featuresAndCfs:
            featureMapping = self.getEncoding(var)
            weights[var] = WeightsSet(featureMapping, offset)
            offset += featureMapping.Size
        return weights, offset

    def setparameters(self, x):
        if any(self.parameters != x):
            self.parameters = x
            self.update()

    def dotproducts_(self, weights, x, parameters):
        results = np.zeros(x.shape[1])
        for w in weights.values():
            results += parameters[w.GetIndices_(x)]
        return results

    def dotproducts(self, weights, x):
        return self.dotproducts_(weights, x, self.parameters)

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

    def predictDF(self, df, pred_col_name: str):
        df = self.transformDf(df)
        return self.predictDFinternal(df, pred_col_name)

    def predict(self, x):
        dotprods = self.dotproducts(self.clickWeights, x) + self.lambdaIntercept
        return 1.0 / (1.0 + np.exp(-dotprods))

    def computeGradientAt(self, x):
        self.setparameters(x)
        return self.computeGradient()

    def update(self):
        raise NotImplementedError

    def predictDFinternal(self, df, pred_col_name: str):
        raise NotImplementedError

    def computeGradient(self):
        raise NotImplementedError

    def computeLoss(self, epsilon=1e-12):
        return 0.0
