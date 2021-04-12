import pandas as pd
import pickle
import pyspark.sql.functions as F
from aggregated_models.diff_priv_noise import GaussianMechanism, LaplaceMechanism
from aggregated_models.featureprojections import (
    ISingleFeatureProjection,
    SingleFeatureProjection,
    CrossFeaturesProjection,
    DataProjection,
    parseCF,
)


class FeaturesSet:
    def __init__(self, features, crossfeaturesStr, df, maxNbModalities=None, singlefeaturesmappings=None):
        self.features = features
        self.crossfeaturesStr = crossfeaturesStr
        self.maxNbModalities = maxNbModalities
        if singlefeaturesmappings is None:
            self.mappings = self.buildSimpleMappings(df)
        else:
            self.mappings = singlefeaturesmappings

        self.crossfeatures = parseCF(features, crossfeaturesStr)
        allfeatures = [f for cf in self.crossfeatures for f in cf]
        if any([f not in features for f in allfeatures]):
            raise Exception("Error: Some cross feature not declared in features list ")
        self.addCrossesMappings()

    def dump(self, handle):
        pickle.dump(self.features, handle)
        pickle.dump(self.crossfeaturesStr, handle)
        pickle.dump(self.maxNbModalities, handle)
        for f in self.features:
            self.mappings[f].dump(handle)

    @staticmethod
    def load(handle, ss=None):
        features = pickle.load(handle)
        crossfeaturesStr = pickle.load(handle)
        maxNbModalities = pickle.load(handle)
        mappings = {}
        for f in features:
            mappings[f] = SingleFeatureProjection.load(handle)
        return FeaturesSet(features, crossfeaturesStr, None, maxNbModalities, mappings)

    def buildSimpleMappings(self, df):
        mappings = {}
        fid = 0
        for var in self.features:
            mapping = SingleFeatureProjection(var, df, fid, self.maxNbModalities)
            mappings[var] = mapping
            fid += 1
        return mappings

    def addCrossesMappings(self):
        mappings = self.mappings
        fid = len(mappings)
        for cf in self.crossfeatures:
            if len(cf) != 2:
                raise Exception("cf of len !=2  not supported yet")
            mapping = CrossFeaturesProjection(mappings[cf[0]], mappings[cf[1]], self.maxNbModalities)
            mappings[mapping.Name] = mapping

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
        if dataframe is None:
            return
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

        self.epsilon0 = epsilon0
        self.delta = delta
        self.removeNegativeValues = removeNegativeValues

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

    def dump(self, handle):
        self.featuresSet.dump(handle)
        pickle.dump({k: v.Data for (k, v) in self.aggDisplays.items()}, handle)
        pickle.dump({k: v.Data for (k, v) in self.aggClicks.items()}, handle)
        pickle.dump(self.label, handle)
        pickle.dump(self.features, handle)
        pickle.dump(self.Nbclicks, handle)
        pickle.dump(self.Nbdisplays, handle)
        pickle.dump(self.epsilon0, handle)
        pickle.dump(self.delta, handle)
        pickle.dump(self.removeNegativeValues, handle)

    @staticmethod
    def load(handle, ss=None):
        self = AggDataset(None, None)

        self.featuresSet = FeaturesSet.load(handle, ss)
        aggDisplaysRaw = pickle.load(handle)
        aggClicksRaw = pickle.load(handle)
        self.label = pickle.load(handle)
        self.features = pickle.load(handle)
        self.Nbclicks = pickle.load(handle)
        self.Nbdisplays = pickle.load(handle)
        self.epsilon0 = pickle.load(handle)
        self.delta = pickle.load(handle)
        self.removeNegativeValues = pickle.load(handle)

        self.aggDisplays = {}
        for k in aggDisplaysRaw:
            feature = self.featuresSet.mappings[k]
            self.aggDisplays[k] = DataProjection(feature, None, k, aggDisplaysRaw[k])
        self.aggClicks = {}
        for k in aggClicksRaw:
            feature = self.featuresSet.mappings[k]
            self.aggClicks[k] = DataProjection(feature, None, k, aggClicksRaw[k])
        return self
