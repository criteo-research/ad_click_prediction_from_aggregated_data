from typing import Dict, List, Optional
import numpy as np
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from aggregated_models.featuremappings import IMapping
from aggregated_models.featureprojections import SingleFeatureProjection
from aggregated_models.featureprojections import (
    DataProjection,
    parseCF,
    ICrossFeaturesProjection,
    ISingleFeatureProjection,
    IProjection,
)
import logging


_log = logging.getLogger(__name__)


class SingleFeatureProjectionSpark(ISingleFeatureProjection):
    """class representing one feature and its set of modalities."""

    def __init__(
        self,
        name: str,
        df: DataFrame,
        fid: int = 0,
        maxNbModalities: Optional[int] = None,
        modalities=None,
        ss=None,
        applyPrehashing: bool = False,
    ):
        """var: name of the feature
        df:  Spark DataFrame
        fid:  index of the feature in np.arrays of shape ( nbfeatures,nbsamples )"""
        self.Name = name
        if applyPrehashing is not None:
            self.hash_col = (F.abs(F.hash(F.col(name))) % F.lit(maxNbModalities)).alias(name)
        else:
            self.hash_col = F.col(name)

        if modalities is None:
            modalitieRows = df.select(self.hash_col).drop_duplicates().orderBy(self.Name).collect()
            modalities = [row[self.Name] for row in modalitieRows]

        from pyspark.sql import SQLContext

        sql_ctx = df.sql_ctx if df is not None else SQLContext(ss)

        self._modalities_broadcast = F.broadcast(
            sql_ctx.createDataFrame([[index, x] for index, x in enumerate(modalities)], schema=("id", self.Name))
        ).persist()

        modalities_count = self._modalities_broadcast.count()  # This is triggered to persist the broadcasted mapping

        if maxNbModalities is not None:
            self.Modulo = maxNbModalities
            if not applyPrehashing:
                self.hash_col = F.col(name) % F.lit(maxNbModalities).alias(name)
        else:
            self.Modulo = modalities_count

        self._default = modalities_count
        self.Size = self._default + 1  # +1 To get a modality for "unobserved"
        self.hashed = True
        self._fid = fid

    def GetModalities(self):
        dico = {row["id"]: row[self.Name] for row in self._modalities_broadcast.collect()}
        return [dico[key] for key in sorted(dico.keys())]

    @staticmethod
    def FromSingleFeatureProjection(f, ss):
        return SingleFeatureProjectionSpark(f.Name, None, f._fid, f._maxNbModalities, f._modalities, ss)

    def ToSingleFeatureProjection(self):
        modalities = self.GetModalities()
        return SingleFeatureProjection(self.Name, None, self._fid, self.Modulo, modalities)

    def dump(self, handle):
        self.ToSingleFeatureProjection().dump(handle)

    # replace each modality of this feature by its index
    def Map(self, df: DataFrame) -> DataFrame:
        return (
            df
            # .withColumn(self.Name, self.hash_col)
            .join(self._modalities_broadcast, on=self.Name, how="left")
            .fillna({"id": self._default})
            .drop(self.Name)
            .withColumnRenamed("id", self.Name)
            .withColumn(self.Name, self.hash_col)
        )

    # df : dataframe
    # col : column of the df to sum
    # return array with, for each modality m of feature: sum( y on rows where feature modality is m)
    def Project(self, df: DataFrame, sum_on: str) -> np.array:
        aggregations = df.select(self.Name, sum_on).groupBy(self.Name).agg(F.sum(sum_on).alias(sum_on)).toPandas()
        projection = np.zeros(self.Size)
        projection[aggregations[self.Name].values] = aggregations[sum_on].values
        return projection


class CrossFeaturesProjectionSpark(ICrossFeaturesProjection):
    """a crossfeature between two single features."""

    def __init__(
        self,
        single_feature_1: ISingleFeatureProjection,
        single_feature_2: ISingleFeatureProjection,
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
