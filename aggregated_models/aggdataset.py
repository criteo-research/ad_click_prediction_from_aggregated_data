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
    GetCfName,
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

            maxNbModalities = self.getMaxNbModalities(var)
            mapping = SingleFeatureProjection(var, df, fid, maxNbModalities)
            mappings[var] = mapping
            fid += 1
        return mappings

    def getMaxNbModalities(self, var):
        if type(self.maxNbModalities) is dict:
            if var in self.maxNbModalities:
                return self.maxNbModalities[var]
            else:
                return self.maxNbModalities["default"]
        return self.maxNbModalities

    def addCrossesMappings(self):
        mappings = self.mappings
        fid = len(mappings)
        for cf in self.crossfeatures:
            if len(cf) != 2:
                raise Exception("cf of len !=2  not supported yet")

            maxNbModalities = self.getMaxNbModalities(GetCfName(cf))
            mapping = CrossFeaturesProjection(mappings[cf[0]], mappings[cf[1]], maxNbModalities)
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

    def fix_fids(self, features_sublist):
        fid = 0
        mappings = self.mappings
        for f in features_sublist:
            mapping = mappings[f]
            mapping._fid = fid
            fid += 1
        for cf in mappings.values():
            if type(cf) is CrossFeaturesProjection:
                cf._fid1 = mappings[cf._v1]._fid
                cf._fid2 = mappings[cf._v2]._fid


class AggDataset:
    _DISPLAY_COL_NAME = "display"

    def __init__(
        self,
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
        if dataframe is None:
            return

        self.epsilon0 = epsilon0
        self.delta = delta
        self.removeNegativeValues = removeNegativeValues
        self.featuresSet = FeaturesSet(features, "*&*", dataframe, maxNbModalities)
        self.columns = [self._DISPLAY_COL_NAME, label] + otherCols
        self.aggregate(dataframe)

        if epsilon0 is not None:
            self.MakeDiffPrivate(epsilon0, delta, removeNegativeValues)
        else:
            self.noiseDistribution = None

    def aggregate(self, dataframe):
        if isinstance(dataframe, pd.DataFrame):
            dataframe[self._DISPLAY_COL_NAME] = 1
        else:
            dataframe = dataframe.withColumn(self._DISPLAY_COL_NAME, F.lit(1))
        self.aggregations = {col: self.featuresSet.Project(dataframe, col) for col in self.columns}
        self.AggregationSums = {col: self.aggregations[col][self.features[0]].Data.sum() for col in self.columns}

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
    def load(handle, ss=None):
        self = AggDataset(None, None)
        self.featuresSet = FeaturesSet.load(handle, ss)
        self.columns = pickle.load(handle)
        self.aggregations = {}
        for col in self.columns:
            aggRaw = pickle.load(handle)
            agg = {}
            for k in aggRaw:
                feature = self.featuresSet.mappings[k]
                agg[k] = DataProjection(feature, None, k, aggRaw[k])
            self.aggregations[col] = agg
        self.AggregationSums = pickle.load(handle)
        self.epsilon0 = pickle.load(handle)
        self.delta = pickle.load(handle)
        self.removeNegativeValues = pickle.load(handle)
        return self

    @staticmethod
    def load_legacy(handle, ss=None):
        self = AggDataset(None, None)

        self.featuresSet = FeaturesSet.load(handle, ss)
        aggDisplaysRaw = pickle.load(handle)
        aggClicksRaw = pickle.load(handle)
        label = pickle.load(handle)
        self.columns = [self._DISPLAY_COL_NAME, label]
        features = pickle.load(handle)
        self.AggregationSums = {}
        self.AggregationSums[self.label] = pickle.load(handle)
        self.AggregationSums[self._DISPLAY_COL_NAME] = pickle.load(handle)

        self.epsilon0 = pickle.load(handle)
        self.delta = pickle.load(handle)
        self.removeNegativeValues = pickle.load(handle)

        self.aggregations = {}
        self.aggregations[self._DISPLAY_COL_NAME] = {}
        for k in aggDisplaysRaw:
            feature = self.featuresSet.mappings[k]
            self.aggDisplays[k] = DataProjection(feature, None, k, aggDisplaysRaw[k])
        self.aggregations[self.label] = {}
        for k in aggClicksRaw:
            feature = self.featuresSet.mappings[k]
            self.aggClicks[k] = DataProjection(feature, None, k, aggClicksRaw[k])
        return self
