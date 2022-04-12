from typing import List
import pandas as pd
import numpy as np
import numba

try:
    from pyspark.sql import functions as F
except:
    pass
from dataclasses import dataclass


@numba.njit
def projectNUMBA(x, y, nbmods):
    mods = np.zeros(nbmods)
    n = len(x)
    for i in np.arange(0, n):
        mods[x[i]] += y[i]
    return mods


def GetCfName(variables):
    return "&".join(sorted(variables))


class IEncoding:
    Size: int

    def Values_(self, x: np.array):
        pass

    def Values(self, df: pd.DataFrame):
        pass

    def SparkCol(self):
        pass

    def ProjectDF(self, df, colname):
        if type(df) is pd.DataFrame:
            return self.ProjectPandasDF(df, colname)
        return self.ProjectSparkDF(df, colname)

    def ProjectPandasDF(self, df, colname):
        y = df[colname].values
        x = self.Values(df)
        return projectNUMBA(x, y, self.Size)

    def ProjectSparkDF(self, df, sum_on):
        col = self.SparkCol()
        dico = df.select(col.alias("toto"), sum_on).groupBy("toto").agg(F.sum(sum_on).alias(sum_on)).rdd.collectAsMap()
        proj = np.zeros(self.Size)
        proj[np.array(list(dico.keys()))] = np.array(list(dico.values()))
        return proj


@dataclass
class SingleFeatureEncoding(IEncoding):
    _fid: int
    Name: str
    Size: int

    def Values_(self, x: np.array):
        return x[self._fid]

    def Values(self, df: pd.DataFrame):
        return df[self.Name].values % self.Size

    def SparkCol(self):
        return F.col(self.Name) % F.lit(self.Size)

    def Project_(self, x: np.ndarray, y: np.array) -> np.array:
        x_values = x[self._fid] % self.Size
        return projectNUMBA(x_values, y, self.Size)

    def __repr__(self):
        return f"{self.Name}({self.Size})"

    @staticmethod
    def FromRawFeatureMapping(fid, rawmapping, maxSize: int = None):
        if maxSize is None:
            maxSize = rawmapping.Size
        return SingleFeatureEncoding(fid, rawmapping.Name, min(maxSize, rawmapping.Size))


@dataclass
class CrossFeatureEncoding(IEncoding):
    _f1: SingleFeatureEncoding
    _f2: SingleFeatureEncoding
    coefV2: int
    Size: int
    hashed: bool = False

    @property
    def _v1(self):
        return self._f1.Name

    @property
    def _v2(self):
        return self._f2.Name

    @property
    def _fid1(self):
        return self._f1._fid

    @property
    def _fid2(self):
        return self._f2._fid

    @property
    def _variables(self):
        return [self._v1, self._v2]

    @property
    def Name(self):
        return GetCfName([f for f in self._variables])

    @property
    def Modulo(self):
        return self.Size

    def Values_(self, x: np.ndarray) -> np.array:
        return (x[self._fid1] + self.coefV2 * x[self._fid2]) % self.Size

    def Values(self, df):
        return (self._f1.Values(df) + self.coefV2 * self._f2.Values(df)) % self.Size

    def SparkCol(self):
        return (self._f1.SparkCol() + self._f2.SparkCol() * F.lit(self.coefV2)) % F.lit(self.Size)

    def Project_(self, x: np.ndarray, y: np.array) -> np.array:
        x_values = self.Values_(x)
        return projectNUMBA(x_values, y, self.Size)

    def __repr__(self):
        if self.hashed:
            return "hash" + self.Name
        return self.Name
    
    def marginalize(self, y: np.ndarray, fname):
        if len(y) != self.Size:
            raise ValueError(f"{this}::marginalize len(y)={len(y)} != Size={self.Size}")
        values = self.modalitiesOtherFeature(fname)
        return projectNUMBA(values, y, len(set(values)) )
    
    def modalitiesOtherFeature(self,fname):
        if fname == self._f1.Name:
            values = self._f2modalities()
        elif fname == self._f2.Name:
            values = self._f1modalities()
        else:
            raise ValueError(f"{this}::marginalize: unknown name {fname}")
        return values
    
    def _f1modalities(self):
        return np.arange(0, self.Size ) % self._f1.Size
    def _f2modalities(self):
        return np.arange(0, self.Size ) // self._f1.Size

    def fromIndepProbas(self, x1: np.ndarray, x2: np.ndarray):
        return np.outer(x1,x2).flatten()

    
    @staticmethod
    def FromSingleFeatureEncodings(f1, f2, maxSize=None):
        size = f1.Size * f2.Size
        if maxSize is None or size <= maxSize:
            return CrossFeatureEncoding(
                f1,
                f2,
                coefV2=f1.Size,
                Size=f1.Size * f2.Size,
                hashed=False,
            )
        else:
            return CrossFeatureEncoding(f1, f2, coefV2=7907, Size=maxSize, hashed=True)
