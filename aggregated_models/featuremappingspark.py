from typing import Dict, List, Optional
import numpy as np
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from aggregated_models.featuremappings import (
    IMapping,
    DataProjection,
    parseCF,
    ICrossFeaturesMapping,
    IFeatureMapping,
)


class FeatureMappingSpark(IFeatureMapping):
    """class representing one feature and its set of modalities."""

    def __init__(self, name: str, df: DataFrame, fid: int = 0, maxNbModalities: Optional[int] = None):
        """var: name of the feature
        df:  Spark DataFrame
        fid:  index of the feature in np.arrays of shape ( nbfeatures,nbsamples )"""
        self.Name = name
        if maxNbModalities is not None:
            self.hash_col = (F.abs(F.hash(F.col(name))) % F.lit(maxNbModalities)).alias(name)
        else:
            self.hash_col = F.col(name)

        modalities = df.select(self.hash_col).drop_duplicates().orderBy(self.Name).collect()

        self._modalities_broadcast = F.broadcast(
            df.sql_ctx.createDataFrame(
                [[index, row[self.Name]] for index, row in enumerate(modalities)], schema=("id", self.Name)
            )
        ).persist()

        modalities_count = self._modalities_broadcast.count()  # This is triggered to persist the broadcasted mapping

        if maxNbModalities is not None:
            self.Modulo = maxNbModalities
        else:
            self.Modulo = modalities_count
        self._default = modalities_count
        self.Size = self._default + 1  # +1 To get a modality for "unobserved"
        self.hashed = True
        self._fid = fid

    # replace each modality of this feature by its index
    def Map(self, df: DataFrame) -> DataFrame:
        return (
            df.withColumn(self.Name, self.hash_col)
            .join(self._modalities_broadcast, on=self.Name, how="left")
            .fillna({"id": self._default})
            .drop(self.Name)
            .withColumnRenamed("id", self.Name)
        )

    # df : dataframe
    # col : column of the df to sum
    # return array with, for each modality m of feature: sum( y on rows where feature modality is m)
    def Project(self, df: DataFrame, sum_on: str) -> np.array:
        aggregations = df.select(self.Name, sum_on).groupBy(self.Name).agg(F.sum(sum_on).alias(sum_on)).toPandas()
        projection = np.zeros(self.Size)
        projection[aggregations[self.Name].values] = aggregations[sum_on].values
        return projection


class CrossFeaturesMappingSpark(ICrossFeaturesMapping):
    """a crossfeature between two single features."""

    def __init__(
        self,
        single_feature_1: IFeatureMapping,
        single_feature_2: IFeatureMapping,
        maxNbModalities: int = None,
    ):
        super().__init__(single_feature_1, single_feature_2, maxNbModalities)

    def Project(self, df: DataFrame, sum_on: str) -> np.array:
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


class FeaturesSetSpark:
    mappings: Dict[str, IMapping]
    features: List[str]
    crossfeatures: List[List[str]]

    def __init__(
        self, features: List[str], crossfeaturesStr: str, df: DataFrame, maxNbModalities: Optional[int] = None
    ):
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
            mappings[var] = FeatureMappingSpark(var, df, fid, self.maxNbModalities)
            fid += 1

        for cf in self.crossfeatures:
            if len(cf) != 2:
                raise Exception("cf of len !=2  not supported yet")
            mapping = CrossFeaturesMappingSpark(mappings[cf[0]], mappings[cf[1]], self.maxNbModalities)
            mappings[mapping.Name] = mapping
        self.mappings = mappings

    def getMapping(self, var: str):
        return self.mappings[var]

    def transformDf(self, df, alsoCrossfeatures=False):
        for var in self.mappings.values():
            if alsoCrossfeatures or type(var) is FeatureMappingSpark:
                df = var.Map(df)
        return df

    def Project(self, train: DataFrame, on: str) -> Dict[str, DataProjection]:
        train = self.transformDf(train)
        projections = {}
        for var in self.mappings:
            projections[var] = DataProjection(self.mappings[var], train, on)
        return projections

    def __repr__(self):
        return ",".join(f.Name for f in self.mappings.values())


class AggDatasetSpark:
    _DISPLAY_COL_NAME = "display"

    def __init__(
        self,
        features,
        train: DataFrame,
        label="click",
        maxNbModalities=None,
    ):
        self.label = label
        self.featuresSet = FeaturesSetSpark(features, "*&*", train, maxNbModalities)
        self.features = self.featuresSet.features
        self.aggClicks = self.featuresSet.Project(train, label)
        self.aggDisplays = self.featuresSet.Project(
            train.withColumn(self._DISPLAY_COL_NAME, F.lit(1)), self._DISPLAY_COL_NAME
        )
        self.Nbclicks = self.aggClicks[features[0]].Data.sum()
        self.Nbdisplays = self.aggDisplays[features[0]].Data.sum()

    def __repr__(self):
        return f"Label:{self.label};featuresSet:{self.featuresSet}"
