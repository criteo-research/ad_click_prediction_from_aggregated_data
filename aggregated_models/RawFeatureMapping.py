import pandas as pd
import numpy as np
from typing import Dict, List, Optional

try:
    from pyspark.sql import DataFrame
    import pyspark.sql.functions as F
except:
    print("failed to load pyspark")
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
        if type(df) is pd.DataFrame:
            return RawFeatureMapping.getModalitiesPandas(df, name)
        else:
            return RawFeatureMapping.getModalitiesSpark(df, name)

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
        if type(df) is pd.DataFrame:
            return self.MapPandas(df)
        return self.MapSpark(df)

    # def MapSpark(self, df: DataFrame) -> DataFrame:
    def MapSpark(self, df):
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

    @staticmethod
    def BuildCtrBuckets(name: str, df, logbase=10, nbStd=1, gaussianStd=10):
        df["d"] = 1
        df = df.groupby(name).sum()
        df = df.reset_index()
        return RawFeatureMapping.BuildCtrBucketsFromAggDf(name, df, logbase, nbStd, gaussianStd)

    @staticmethod
    def BuildCtrBucketsFromAggDf(name: str, df, logbase, nbStd, gaussianStd):
        dicoModalityToId = RawFeatureMapping.ctrBucketsMapping(name, df, logbase, nbStd, gaussianStd)
        return RawFeatureMapping(name, dicoModalityToId)

    @staticmethod
    def ctrBucketsMapping(f, df, logbase, nbStd, gaussianStd):
        def getThreesholds(gaussianStd, maxN=1_000_000, nbStd=1.0):
            ts = []
            c = 1 + gaussianStd
            while c < maxN:
                std = np.sqrt(c) + gaussianStd
                c += nbStd * std
                ts.append(c)
            return np.array(ts)

        allThreeshold = getThreesholds(gaussianStd, df.click.max() * logbase, nbStd)
        prior = df.click.sum() / df.d.sum()
        df["roundedD"] = roundedD = logbase ** (1 + np.floor(np.log10(df.d) / np.log10(logbase)))
        d = df.d.values
        c = df.click.values
        df["ctr"] = ctr = (c + prior) / (d + 1)
        c_at_roundedD = ctr * roundedD
        import bisect

        df["ctrBucketId"] = [bisect.bisect(allThreeshold, x) for x in c_at_roundedD]
        # priorStd = np.sqrt( prior * (1-prior ) *  roundedD )/roundedD
        # priorStd *= nbStd
        # df["ctrBucketId"] = np.floor (ctr/priorStd) * priorStd
        df["key"] = list(zip(df["roundedD"].values, df["ctrBucketId"].values))
        allkeys = sorted(set(df["key"].values))
        len(allkeys)
        keysDico = {k: i for i, k in enumerate(allkeys)}
        df["newid"] = [keysDico[k] for k in df["key"].values]
        dicoOldModalityToNewModality = {old: new for old, new in zip(df[f], df["newid"])}
        return dicoOldModalityToNewModality


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
                toto
        return df

    def __repr__(self):
        return ",".join(f.Name for f in self.rawmappings.values())
