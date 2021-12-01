import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import pickle
from dataclasses import dataclass


@dataclass
class RawFeatureMapping:
    """class representing one feature and its set of modalities."""

    Name: str
    _dicoModalityToId: Dict[int, int]

    def __post_init__(self):
        self._modalities_broadcast = None
        self._default = max(self._dicoModalityToId.values()) + 1
        self.Size = self._default + 1

    @staticmethod
    def FromDF(name: str, df):
        modalities = RawFeatureMapping.getModalities(df, name)
        dicoModalityToId = {m: i for i, m in enumerate(modalities)}
        return RawFeatureMapping(name, dicoModalityToId)

    # list of modalities observed in df
    @staticmethod
    def getModalities(df, name):
        if type(df) is DataFrame:
            return RawFeatureMapping.getModalitiesSpark(df, name)
        else:
            return RawFeatureMapping.getModalitiesPandas(df, name)

    @staticmethod
    def getModalitiesPandas(df, name):
        return [int(x) for x in (sorted(set(df[name].values)))]

    @staticmethod
    def getModalitiesSpark(df, name):
        modalitieRows = df.select(name).drop_duplicates().orderBy(name).collect()
        modalities = list([row[name] for row in modalitieRows])
        return modalities

    def dump(self, handle):
        pickle.dump(self.Name, handle)
        pickle.dump(self._dicoModalityToId, handle)

    @staticmethod
    def load(handle, ss=None):
        name = pickle.load(handle)
        modalities = pickle.load(handle)
        return RawFeatureMapping(name, modalities)

    def spark_col(self):
        return F.col(self.Name).alias(self.Name)

    def setBroadCast(self, sql_ctx):
        if self._modalities_broadcast is not None:
            return
        self._modalities_broadcast = F.broadcast(
            sql_ctx.createDataFrame(
                [[int(newMod), int(oldMod)] for oldMod, newMod in _dicoModalityToId.items()], schema=("id", self.Name)
            )
        ).persist()

    # replace initial modalities of features by modality index
    def Map(self, df):
        if type(df) is DataFrame:
            return self.MapSpark(df)
        return self.MapPandas(df)

    def MapSpark(self, df: DataFrame) -> DataFrame:
        self.setBroadCast(df.sql_ctx)
        return (
            df.join(self._modalities_broadcast, on=self.Name, how="left")
            .fillna({"id": self._default})
            .drop(self.Name)
            .withColumnRenamed("id", self.Name)
            .withColumn(self.Name, self.spark_col())
        )

    def MapPandas(self, df) -> pd.DataFrame:
        df[self.Name] = df[self.Name].apply(lambda x: self._dicoModalityToId.get(x, self._default))
        return df

    def Values(self, df: pd.DataFrame):
        return df[self.Name].values


class RawFeaturesSet:
    def __init__(self, features, rawmappings):
        self.features = features
        self.rawmappings = rawmappings

    @staticmethod
    def FromDF(features, df):
        rawmappings = {f: RawFeatureMapping.FromDF(f, df) for f in features}
        return RawFeaturesSet(features, rawmappings)

    def dump(self, handle):
        pickle.dump(self.features, handle)
        for f in self.features:
            self.rawmappings[f].dump(handle)

    @staticmethod
    def load(handle, ss=None):
        features = pickle.load(handle)
        mappings = {}
        for f in features:
            mappings[f] = RawFeatureMapping.load(handle)
        return RawFeaturesSet(features, mappings)

    def Map(self, df):
        if isinstance(df, pd.DataFrame):
            df = df.copy()
        for var in self.rawmappings.values():
            if var.Name in df.columns:
                df = var.Map(df)
            else:
                print("warning:: RawFeaturesSet.Map :: feature " + var.Name + " not found in df")
        return df

    def __repr__(self):
        return ",".join(f.Name for f in self.rawmappings.values())
