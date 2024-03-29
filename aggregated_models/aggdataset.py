from dataclasses import dataclass
import pandas as pd
import pickle

try:
    import pyspark.sql.functions as F
except:
    print("failed to load pyspark")
from typing import Dict, List

from aggregated_models.diff_priv_noise import GaussianMechanism, LaplaceMechanism
from aggregated_models.CrossFeaturesSet import CrossFeaturesSet

from aggregated_models.FeatureEncodings import *


@dataclass
class DataProjection:
    feature: IEncoding
    colName: str
    Data: np.array

    @staticmethod
    def FromDF(feature: IEncoding, df: pd.DataFrame, colName: str):
        data = feature.ProjectDF(df, colName)
        return DataProjection(feature, colName, data)

    @staticmethod
    def Zeros(feature: IEncoding, colName: str):
        data = np.zeros(feature.Size)
        return DataProjection(feature, colName, data)

    def _build(self, feature, data, colName):
        self.feature = feature
        self.colName = colName
        self.Data = data

    def __repr__(self):
        return f"Projection {self.colName} on {self.feature}"

    def dump(self, handle):
        self.feature.dump(handle)
        pickle.dump(self.colName, handle)
        pickle.dump(self.Data, handle)

    @staticmethod
    def load(handle):
        feature = SingleFeatureProjection.load(handle)
        colName = pickle.load(handle)
        data = pickle.load(handle)
        return DataProjection(feature, None, colName, data)

    def copy0(self):
        return DataProjection(self.feature, None, self.colName, np.zeros(len(self.Data)))

    def Add(self, df):
        self.Data += self.feature.ProjectDF(df, self.colName)


_DISPLAY_COL_NAME = "display"


@dataclass
class AggDataset:
    featuresSet: CrossFeaturesSet
    columns: List[str]
    epsilon0: float = None
    delta: float = None
    removeNegativeValues: bool = False

    def __post_init__(self):
        self._DISPLAY_COL_NAME = self.columns[0]
        self.aggregations = {}
        for col in self.columns:
            self.aggregations[col] = {
                encoding.Name: DataProjection.Zeros(encoding, col) for encoding in self.featuresSet.encodings.values()
            }
        self.AggregationSums = {col: 0 for col in self.columns}

    @staticmethod
    def FromDF(
        dataframe,
        features,
        cf="*&*",
        label="click",
        otherCols=[],
        epsilon0=None,
        delta=None,
        removeNegativeValues=False,
        maxNbModalities=None,
    ):
        featuresSet = CrossFeaturesSet.FromDf(dataframe, features, maxNbModalities, "*&*")
        columns = [_DISPLAY_COL_NAME, label] + otherCols
        self = AggDataset(featuresSet, columns, epsilon0, delta, removeNegativeValues)
        self.aggregate(dataframe)
        if epsilon0 is not None:
            self.MakeDiffPrivate(epsilon0, delta, removeNegativeValues)
        else:
            self.noiseDistribution = None

        return self

    def aggregate(self, dataframe):
        if isinstance(dataframe, pd.DataFrame):
            dataframe[self._DISPLAY_COL_NAME] = 1
        else:
            dataframe = dataframe.withColumn(self._DISPLAY_COL_NAME, F.lit(1))

        dataframe = self.featuresSet.transformDf(dataframe)
        for col in self.columns:
            for p in self.aggregations[col].values():
                p.Add(dataframe)
            self.AggregationSums[col] = self.aggregations[col][self.features[0]].Data.sum()

    @property
    def features(self):
        return self.featuresSet.features

    @property
    def label(self):
        return self.columns[1]

    @property
    def aggDisplays(self):
        return self.aggregations[self._DISPLAY_COL_NAME]

    @property
    def aggClicks(self):
        return self.aggregations[self.label]

    @property
    def Nbdisplays(self):
        return self.AggregationSums[self._DISPLAY_COL_NAME]

    @property
    def Nbclicks(self):
        return self.AggregationSums[self.label]

    def toDFs(self):
        dfs = {}
        for var, p in self.aggDisplays.items():
            dfs[var] = p.toDF()
            if var in self.aggClicks:
                dfs[var][self.label] = self.aggClicks[var].Data
        return dfs

    def __repr__(self):
        return f"Label:{self.label};featuresSet:{self.featuresSet}"

    def MakeDiffPrivate(self, epsilon0=1.0, delta=None, removeNegativeValues=False):
        self.epsilon0 = epsilon0  # garanteed epsilon
        self.delta = delta
        nbQuerries = len(self.aggDisplays) * len(self.columns)
        if delta is None:
            self.mechanism = LaplaceMechanism(epsilon0, nbQuerries)
        else:
            self.mechanism = GaussianMechanism(epsilon0, delta, nbQuerries)
        print(self.mechanism)
        self.noiseDistribution = self.mechanism.getNoise()

        for agg in self.aggregations.values():
            for proj in agg.values():
                n = len(proj.Data)
                proj.Data += self.noiseDistribution.Sample(n)
                if removeNegativeValues:
                    proj.Data[proj.Data < 0] = 0
        return self

    def dump(self, handle):
        self.featuresSet.dump(handle)
        pickle.dump(self.columns, handle)
        for agg in self.aggregations.values():
            pickle.dump({k: v.Data for (k, v) in agg.items()}, handle)
        pickle.dump(self.AggregationSums, handle)
        pickle.dump(self.epsilon0, handle)
        pickle.dump(self.delta, handle)
        pickle.dump(self.removeNegativeValues, handle)

    @staticmethod
    def load(handle):
        featuresSet = CrossFeaturesSet.load(handle)
        columns = pickle.load(handle)
        aggregations = {}
        for col in columns:
            aggRaw = pickle.load(handle)
            agg = {}
            for k in aggRaw:
                feature = featuresSet.encodings[k]
                agg[k] = DataProjection(feature, col, aggRaw[k])
            aggregations[col] = agg
        AggregationSums = pickle.load(handle)
        epsilon0 = pickle.load(handle)
        delta = pickle.load(handle)
        removeNegativeValues = pickle.load(handle)

        self = AggDataset(featuresSet, columns, epsilon0, delta, removeNegativeValues)
        self.aggregations = aggregations
        self.AggregationSums = AggregationSums

        return self
