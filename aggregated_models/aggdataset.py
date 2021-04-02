import pandas as pd
import pyspark.sql.functions as F
from aggregated_models.diff_priv_noise import GaussianMechanism, LaplaceMechanism
from aggregated_models.featureprojections import (
    ISingleFeatureProjection,
    SingleFeatureProjection,
    CrossFeaturesProjection,
    DataProjection,
    parseCF,
)
from aggregated_models.featureprojectionspark import (
    SingleFeatureProjectionSpark,
    CrossFeaturesProjectionSpark,
)


class FeaturesSet:
    def __init__(self, features, crossfeaturesStr, df, maxNbModalities=None):
        self.features = features
        self.crossfeatures = parseCF(features, crossfeaturesStr)
        self.maxNbModalities = maxNbModalities

        allfeatures = [f for cf in self.crossfeatures for f in cf]
        if any([f not in features for f in allfeatures]):
            raise Exception("Error: Some cross feature not declared in features list ")
        self.buildFeaturesMapping(df)

    def buildFeaturesMapping(self, df):
        mappings = {}
        fid = 0
        for var in self.features:
            if isinstance(df, pd.DataFrame):
                mapping = SingleFeatureProjection(var, df, fid, self.maxNbModalities)
            else:
                mapping = SingleFeatureProjectionSpark(var, df, fid, self.maxNbModalities)
            mappings[var] = mapping
            fid += 1

        for cf in self.crossfeatures:
            if len(cf) != 2:
                raise Exception("cf of len !=2  not supported yet")
            if isinstance(df, pd.DataFrame):
                mapping = CrossFeaturesProjection(mappings[cf[0]], mappings[cf[1]], self.maxNbModalities)
            else:
                mapping = CrossFeaturesProjectionSpark(mappings[cf[0]], mappings[cf[1]], self.maxNbModalities)
            mappings[mapping.Name] = mapping
        self.mappings = mappings

    def getMapping(self, var):
        return self.mappings[var].get_feature_mapping()

    def transformDf(self, df, alsoCrossfeatures=False):
        if isinstance(df, pd.DataFrame):
            df = df.copy()
        for var in self.mappings.values():
            if alsoCrossfeatures or isinstance(var, ISingleFeatureProjection):
                df = var.Map(df)
        return df

    def Project(self, dataframe, column):
        dataframe = self.transformDf(dataframe)
        projections = {}
        for var in self.mappings:
            projections[var] = DataProjection(self.mappings[var], dataframe, column)
        return projections

    def __repr__(self):
        return ",".join(f.Name for f in self.mappings.values())


class AggDataset:
    _DISPLAY_COL_NAME = "display"

    def __init__(
        self,
        dataframe,
        features,
        cf="*&*",
        label="click",
        epsilon0=None,
        delta=None,
        removeNegativeValues=False,
        maxNbModalities=None,
    ):
        self.label = label
        self.featuresSet = FeaturesSet(features, "*&*", dataframe, maxNbModalities)
        self.features = self.featuresSet.features
        self.aggClicks = self.featuresSet.Project(dataframe, label)
        if isinstance(dataframe, pd.DataFrame):
            dataframe[self._DISPLAY_COL_NAME] = 1
        else:
            dataframe = dataframe.withColumn(self._DISPLAY_COL_NAME, F.lit(1))
        self.aggDisplays = self.featuresSet.Project(dataframe, self._DISPLAY_COL_NAME)
        self.Nbclicks = self.aggClicks[features[0]].Data.sum()
        self.Nbdisplays = self.aggDisplays[features[0]].Data.sum()

        if epsilon0 is not None:
            self.MakeDiffPrivate(epsilon0, delta, removeNegativeValues)
        else:
            self.noiseDistribution = None

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
        nbQuerries = len(self.aggDisplays)
        if delta is None:
            self.mechanism = LaplaceMechanism(epsilon0, nbQuerries)
        else:
            self.mechanism = GaussianMechanism(epsilon0, delta, nbQuerries)
        print(self.mechanism)
        self.noiseDistribution = self.mechanism.getNoise()

        for proj in self.aggClicks.values():
            n = len(proj.Data)
            proj.Data += self.noiseDistribution.Sample(n)
            if removeNegativeValues:
                proj.Data[proj.Data < 0] = 0
        for proj in self.aggDisplays.values():
            n = len(proj.Data)
            proj.Data += self.noiseDistribution.Sample(n)
            if removeNegativeValues:
                proj.Data[proj.Data < 0] = 0
        return self
