from typing import List
from numpy.lib.arraypad import pad
import pandas as pd
import numpy as np
import numba
from typing import Dict, List, Optional
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from aggregated_models import noiseDistributions
from aggregated_models.diff_priv_noise import GaussianMechanism, LaplaceMechanism
from aggregated_models.featuremappings import (
    projectNUMBA,
    GetCfName,
    IFeature,
    SingleFeatureMapping,
    CrossFeaturesMapping,
)
import logging
import pickle


_log = logging.getLogger(__name__)


class DataProjection:
    def __init__(self, feature, df, colName, data=None):
        if data is None:
            data = feature.Project(df, colName)
        self._build(feature, data, colName)

    def _build(self, feature, data, colName):
        self.feature = feature
        self.colName = colName
        self.Data = data

    def __repr__(self):
        return f"Projection {self.colName} on {self.feature}"

    def toDF(self):
        df = self.feature.toDF()
        df[self.colName] = self.Data
        return df

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
        self.Data += self.feature.Project(df, self.colName)


def parseCFNames(features, crossfeaturesStr):
    cfs = parseCF(features, crossfeaturesStr)
    return [GetCfName(cf) for cf in cfs]


def parseCF(features, crossfeaturesStr):
    cfs = []
    crossfeaturesStr = crossfeaturesStr.split("|")
    for cfStr in crossfeaturesStr:
        cfFeatures = cfStr.split("&")
        nbWildcards = len([f for f in cfFeatures if f == "*"])
        cfFeatures = [[f for f in cfFeatures if not f == "*"]]
        for i in range(0, nbWildcards):
            cfFeatures = [cf + [v] for v in features for cf in cfFeatures]
        cfFeatures = [sorted(f) for f in cfFeatures]
        cfs += cfFeatures
    cfs = [list(sorted(set(cf))) for cf in cfs]
    cfs = [cf for cf in cfs if len(cf) == 2]
    # remove duplicates
    dicoCfsStr = {}
    for cf in cfs:
        s = "&".join([str(f) for f in cf])
        dicoCfsStr[s] = cf
    cfs = [cf for cf in dicoCfsStr.values()]
    return cfs


class IProjection(IFeature):
    def get_feature_mapping(self):
        pass


class ISingleFeatureProjection(IProjection):
    _fid: int
    Modulo: int

    def get_feature_mapping(self):
        return SingleFeatureMapping(self.Name, self._fid, self.Size, self.Modulo)


class SingleFeatureProjection(ISingleFeatureProjection):
    """class representing one feature and its set of modalities."""

    def __init__(self, name: str, df: pd.DataFrame, fid: int = 0, maxNbModalities: int = None, modalities=None):
        """var: name of the feature
        df:  pd.DataFrame containing this feature
        fid:  index of the feature in np.arrays of shape ( nbfeatures,nbsamples )"""
        if modalities is None:
            modalities = SingleFeatureProjection.getModalities(df, name)
        self._build(name, modalities, fid, maxNbModalities)

    # list of modalities observed in df
    @staticmethod
    def getModalities(df, name):
        if type(df) is DataFrame:
            return SingleFeatureProjection.getModalitiesSpark(df, name)
        else:
            return SingleFeatureProjection.getModalitiesPandas(df, name)

    @staticmethod
    def getModalitiesPandas(df, name):
        return list(sorted(set(df[name].values)))

    @staticmethod
    def getModalitiesSpark(df, name):
        modalitieRows = df.select(name).drop_duplicates().orderBy(name).collect()
        modalities = list([row[name] for row in modalitieRows])
        return modalities

    def _build(self, name: str, modalities, fid: int = 0, maxNbModalities: int = None):
        self.Name = name
        self._modalities = modalities
        self._maxNbModalities = maxNbModalities
        self._dicoModalityToId = {m: i for i, m in enumerate(self._modalities)}  # assigning an id to each modality
        self._default = len(self._modalities)  # assigning an id for eventual modalities for observed in df
        self.Size = len(self._modalities) + 1  # +1 To get a modality for "unobserved"
        self._fid = fid

        if maxNbModalities is None or len(self._modalities) < maxNbModalities:
            self._dicoModalityToId = {m: i for i, m in enumerate(self._modalities)}  # assigning an id to each modality
            self._default = len(self._modalities)  # assigning an id for eventual modalities for observed in dataframe
            self.Size = len(self._modalities) + 1  # +1 To get a modality for "unobserved"
            self.Modulo = self.Size + 1  # to implement some hashing later
            self.hashed = False
        else:
            self.Modulo = maxNbModalities  # to implement some hashing later
            self._dicoModalityToId = {
                m: i % self.Modulo for i, m in enumerate(self._modalities)
            }  # assigning an id to each modality
            self._default = self.Modulo  # assigning an id for eventual modalities for observed in dataframe
            self.Size = self._default + 1  # +1 To get a modality for "unobserved"
            self.hashed = True
        self._modalities_broadcast = None

    def spark_col(self):
        return F.col(self.Name) % F.lit(self.Modulo).alias(self.Name)

    def setBroadCast(self, sql_ctx):

        modalitiesInt = [int(x) for x in self._modalities]  # Making sure with have 'int', not 'numpy.int64'
        self._modalities_broadcast = F.broadcast(
            sql_ctx.createDataFrame([[index, x] for index, x in enumerate(modalitiesInt)], schema=("id", self.Name))
        ).persist()

    # replace initial modalities of features by modality index
    def Map(self, df):
        if type(df) is DataFrame:
            return self.MapSpark(df)
        return self.MapPandas(df)

    def MapSpark(self, df: DataFrame) -> DataFrame:
        if self._modalities_broadcast is None:
            self.setBroadCast(df.sql_ctx)
        return (
            df.join(self._modalities_broadcast, on=self.Name, how="left")
            .fillna({"id": self._default})
            .drop(self.Name)
            .withColumnRenamed("id", self.Name)
            .withColumn(self.Name, self.spark_col())
        )

    def MapPandas(self, df):
        df[self.Name] = df[self.Name].apply(lambda x: self._dicoModalityToId.get(x, self._default))
        return df

    def Values(self, df: pd.DataFrame):
        return df[self.Name].values

    # df : dataframe
    # col : column of the df to sum
    # return array with, for each modality m of feature: sum( y on rows where feature modality is m)
    def Project(self, df, col):
        if type(df) is DataFrame:
            return self.ProjectSpark(df, col)
        else:
            return self.ProjectPandas(df, col)

    def ProjectSpark(self, df: pd.DataFrame, sum_on: str):
        aggregations = df.select(self.Name, sum_on).groupBy(self.Name).agg(F.sum(sum_on).alias(sum_on)).toPandas()
        projection = np.zeros(self.Size)
        projection[aggregations[self.Name].values] = aggregations[sum_on].values
        return projection

    def ProjectPandas(self, df: pd.DataFrame, col):
        groupedDF = df[[self.Name, col]].groupby(self.Name).sum()
        data = np.zeros(self.Size)
        data[groupedDF.index] = groupedDF[col]
        return data

    def dump(sf, handle):
        pickle.dump(sf.Name, handle)
        pickle.dump(sf._modalities, handle)
        pickle.dump(sf._fid, handle)
        pickle.dump(sf.Modulo, handle)

    @staticmethod
    def load(handle):
        name = pickle.load(handle)
        modalities = pickle.load(handle)
        fid = pickle.load(handle)
        modulo = pickle.load(handle)
        sf = SingleFeatureProjection(name, None, fid, modulo, modalities)
        return sf


class ICrossFeaturesProjection(IFeature):
    def __init__(
        self,
        singleFeature1: ISingleFeatureProjection,
        singleFeature2: ISingleFeatureProjection,
        maxNbModalities: int = None,
    ):
        self._variables = [singleFeature1, singleFeature2]
        self._v1 = self._variables[0].Name
        self._v2 = self._variables[1].Name
        self._fid1 = self._variables[0]._fid
        self._fid2 = self._variables[1]._fid
        self.Name = GetCfName([f.Name for f in self._variables])

        self.Size = singleFeature1.Size * singleFeature2.Size
        if maxNbModalities is None or self.Size < maxNbModalities:
            self.coefV2 = singleFeature1.Size
            self.Modulo = self.Size + 1
            self.hashed = False
        else:
            self.Size = maxNbModalities
            self.Modulo = maxNbModalities
            self.hashed = True
            self.coefV2 = 7907

    def get_feature_mapping(self):
        return CrossFeaturesMapping(
            self._v1,
            self._v2,
            self._fid1,
            self._fid2,
            self.Size,
            self.coefV2,
            self.Modulo,
            self.hashed,
        )


class CrossFeaturesProjection(ICrossFeaturesProjection):
    """a crossfeature between two single features."""

    def __init__(
        self,
        singleFeature1: SingleFeatureProjection,
        singleFeature2: SingleFeatureProjection,
        maxNbModalities: int = None,
    ):
        super().__init__(singleFeature1, singleFeature2, maxNbModalities)

    def Values(self, df):
        return (df[self._v1].values + self.coefV2 * df[self._v2].values) % self.Modulo

    def Project(self, df, col):
        if type(df) is DataFrame:
            return self.ProjectSpark(df, col)
        else:
            return self.ProjectPandas(df, col)

    def ProjectPandas(self, df, col):
        x = self.Values(df)
        y = df[col].values
        if isinstance(x, np.int64):
            raise Exception(f"x:{x},y:{y},fid:{self._fid1}*{self._fid2},size:{self.Size}")
        return projectNUMBA(x, y, self.Size)

    def ProjectSpark(self, df: DataFrame, sum_on: str) -> np.array:
        aggregations = (
            df.select(
                ((F.col(self._v1) + F.lit(self.coefV2) * F.col(self._v2)) % F.lit(self.Modulo)).alias(self.Name), sum_on
            )
            .groupBy(self.Name)
            .agg(F.sum(sum_on).alias(sum_on))
            .toPandas()
        )
        projection = np.zeros(self.Size)
        projection[aggregations[self.Name].values] = aggregations[sum_on].values
        return projection

    def Map(self, df):
        df[self.Name] = self.Values(df)
        return df
