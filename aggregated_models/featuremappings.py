from typing import List
from numpy.lib.arraypad import pad
import pandas as pd
import numpy as np
import numba
from aggregated_models import noiseDistributions
from aggregated_models.diff_priv_noise import GaussianMechanism, LaplaceMechanism


@numba.njit
def projectNUMBA(x, y, nbmods):
    mods = np.zeros(nbmods)
    n = len(x)
    for i in np.arange(0, n):
        mods[x[i]] += y[i]
    return mods


def GetCfName(variables):
    return "&".join(sorted(variables))


class IFeature:
    Size: int
    Name: str
    Modulo: int


class IMapping(IFeature):
    # returning feature values from a np.array of shape (nbfeatures, nbsamples)
    def Values_(self, x: np.array):
        pass

    # x : np.array of shape (nbfeatures, nbsamples)
    # y : array of len( nbsamples )
    # return array with, for each modality m of feature: sum( y on samples where feature modality is m)
    def Project_(self, x: np.ndarray, y: np.array) -> np.array:
        pass


class SingleFeatureMapping(IMapping):
    _fid: int

    def __init__(self, name, fid, size, modulo):
        self.Name = name
        self._fid = fid
        self.Size = size
        self.Modulo = modulo

    def Values_(self, x: np.array):
        return x[self._fid]

    def Values(self, df: pd.DataFrame):
        return df[self.Name].values

    def Project_(self, x: np.ndarray, y: np.array) -> np.array:
        x_values = x[self._fid]
        return projectNUMBA(x_values, y, self.Size)

    def __repr__(self):
        return f"{self.Name}({self.Size})"

    def toDF(self):
        return pd.DataFrame({self.Name: np.arange(0, self.Size)})


class CrossFeaturesMapping(IMapping):
    _fid1: int
    _fid2: int
    coefV2: int
    Size: int
    Name: str
    hashed: bool
    _variables: List[str]

    def __init__(
        self,
        feature_name_1: str,
        feature_name_2: str,
        feature_id_1: int,
        feature_id_2: int,
        size: int,
        coefV2: int,
        modulo: int,
        hashed: bool,
    ):
        self._variables = [feature_name_1, feature_name_2]
        self._v1 = feature_name_1
        self._v2 = feature_name_2
        self._fid1 = feature_id_1
        self._fid2 = feature_id_2
        self.Name = GetCfName([f for f in self._variables])

        self.Size = size
        self.coefV2 = coefV2
        self.Modulo = modulo
        self.hashed = hashed

    def Values_(self, x: np.array) -> int:
        return (x[self._fid1] + self.coefV2 * x[self._fid2]) % self.Modulo

    def Values(self, df):
        return (df[self._v1].values + self.coefV2 * df[self._v2].values) % self.Modulo

    def Project_(self, x: np.ndarray, y: np.array) -> np.array:
        x_values = self.Values_(x)
        return projectNUMBA(x_values, y, self.Size)

    def __repr__(self):
        s = "x".join([n for n in self._variables])
        if self.hashed:
            return "h(" + s + ")"
        return s

    def ValueVar1(self, x):
        return x % self.coefV2

    def ValueVar2(self, x):
        return (x - x % self.coefV2) / self.coefV2

    def toDF(self):
        df = pd.DataFrame({self.Name: np.arange(0, self.Size)})
        df[self._v1] = self.ValueVar1(df[self.Name])
        df[self._v2] = self.ValueVar2(df[self.Name]).astype(int)
        return df
