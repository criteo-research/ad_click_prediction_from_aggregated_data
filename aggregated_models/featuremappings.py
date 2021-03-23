from typing import List
from numpy.lib.arraypad import pad
import pandas as pd
import numpy as np
import numba
from aggregated_models import noiseDistributions
from aggregated_models.diff_priv_noise import GaussianMechanism, LaplaceMechanism


class DataProjection:
    def __init__(self, feature, df, colName):
        self.feature = feature
        self.colName = colName
        self.Data = feature.Project(df, colName)

    def __repr__(self):
        return f"Projection {self.colName} on {self.feature}"

    def toDF(self):
        df = self.feature.toDF()
        df[self.colName] = self.Data
        return df


@numba.njit
def projectNUMBA(x, y, nbmods):
    mods = np.zeros(nbmods)
    n = len(x)
    for i in np.arange(0, n):
        mods[x[i]] += y[i]
    return mods


def GetCfName(variables):
    return "&".join(sorted(variables))


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


class IMapping:
    Size: int
    Name: str

    # returning feature values from a np.array of shape (nbfeatures, nbsamples)
    def Values_(self, x: np.array):
        pass

    # x : np.array of shape (nbfeatures, nbsamples)
    # y : array of len( nbsamples )
    # return array with, for each modality m of feature: sum( y on samples where feature modality is m)
    def Project_(self, x: np.ndarray, y: np.array) -> np.array:
        pass


class IFeatureMapping(IMapping):
    _fid: int

    def Values_(self, x: np.array):
        return x[self._fid]

    def Project_(self, x: np.ndarray, y: np.array) -> np.array:
        x_values = x[self._fid]
        return projectNUMBA(x_values, y, self.Size)

    def __repr__(self):
        return f"{self.Name}({self.Size})"

    def toDF(self):
        return pd.DataFrame({self.Name: np.arange(0, self.Size)})


class FeatureMapping(IFeatureMapping):
    """class representing one feature and its set of modalities."""

    def __init__(self, name: str, df: pd.DataFrame, fid: int = 0, maxNbModalities: int = None):
        """var: name of the feature
        df:  pd.DataFrame containing this feature
        fid:  index of the feature in np.arrays of shape ( nbfeatures,nbsamples )"""
        self.Name = name
        self._modalities = np.array(sorted(set(df[name].values)))  # list of modalities observed in df
        self._dicoModalityToId = {m: i for i, m in enumerate(self._modalities)}  # assigning an id to each modality
        self._default = len(self._modalities)  # assigning an id for eventual modalities for observed in df
        self.Size = len(self._modalities) + 1  # +1 To get a modality for "unobserved"
        self._fid = fid

        if maxNbModalities is None or len(self._modalities) < maxNbModalities:
            self._dicoModalityToId = {m: i for i, m in enumerate(self._modalities)}  # assigning an id to each modality
            self._default = len(self._modalities)  # assigning an id for eventual modalities for observed in train
            self.Size = len(self._modalities) + 1  # +1 To get a modality for "unobserved"
            self.Modulo = self.Size + 1  # to implement some hashing later
            self.hashed = False
        else:
            self.Modulo = maxNbModalities  # to implement some hashing later
            self._dicoModalityToId = {
                m: i % self.Modulo for i, m in enumerate(self._modalities)
            }  # assigning an id to each modality
            self._default = self.Modulo  # assigning an id for eventual modalities for observed in train
            self.Size = self._default + 1  # +1 To get a modality for "unobserved"
            self.hashed = True

    # replace initial modalities of features by modality index
    def Map(self, df):
        df[self.Name] = df[self.Name].apply(lambda x: self._dicoModalityToId.get(x, self._default))
        return df

    def Values(self, df: pd.DataFrame):
        return df[self.Name].values

    # df : dataframe
    # col : column of the df to sum
    # return array with, for each modality m of feature: sum( y on rows where feature modality is m)
    def Project(self, df: pd.DataFrame, col):
        groupedDF = df[[self.Name, col]].groupby(self.Name).sum()
        data = np.zeros(self.Size)
        data[groupedDF.index] = groupedDF[col]
        return data


class ICrossFeaturesMapping(IMapping):
    _fid1: int
    _fid2: int
    coefV2: int
    Modulo: int
    Size: int
    Name: str
    hashed: bool
    _variables: List[IFeatureMapping]

    def __init__(
        self,
        singleFeature1: IFeatureMapping,
        singleFeature2: IFeatureMapping,
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

    def Values_(self, x: np.array) -> int:
        return (x[self._fid1] + self.coefV2 * x[self._fid2]) % self.Modulo

    def Project_(self, x: np.ndarray, y: np.array) -> np.array:
        x_values = self.Values_(x)
        return projectNUMBA(x_values, y, self.Size)

    def __repr__(self):
        s = "x".join([str(x) for x in self._variables])
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


class CrossFeaturesMapping(ICrossFeaturesMapping):
    """a crossfeature between two single features."""

    def __init__(
        self,
        singleFeature1: FeatureMapping,
        singleFeature2: FeatureMapping,
        maxNbModalities: int = None,
    ):
        super().__init__(singleFeature1, singleFeature2, maxNbModalities)

    def Values(self, df):
        return (df[self._v1].values + self.coefV2 * df[self._v2].values) % self.Modulo

    def Project(self, df, col):
        x = self.Values(df)
        y = df[col].values
        if isinstance(x, np.int64):
            raise Exception(f"x:{x},y:{y},fid:{self._fid},siz:{self.Size}")
        return projectNUMBA(x, y, self.Size)

    def Map(self, df):
        df[self.Name] = self.Values(df)
        return df


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
            mappings[var] = FeatureMapping(var, df, fid, self.maxNbModalities)
            fid += 1

        for cf in self.crossfeatures:
            if len(cf) != 2:
                raise Exception("cf of len !=2  not supported yet")
            mapping = CrossFeaturesMapping(mappings[cf[0]], mappings[cf[1]], self.maxNbModalities)
            mappings[mapping.Name] = mapping
        self.mappings = mappings

    def getMapping(self, var):
        return self.mappings[var]

    def transformDf(self, df, alsoCrossfeatures=False):
        df = df.copy()
        for var in self.mappings.values():
            if alsoCrossfeatures or type(var) is FeatureMapping:
                df = var.Map(df)
        return df

    def Project(self, train, column):
        train = self.transformDf(train)
        projections = {}
        for var in self.mappings:
            projections[var] = DataProjection(self.mappings[var], train, column)
        return projections

    def __repr__(self):
        return ",".join(f.Name for f in self.mappings.values())


class AggDataset:
    def __init__(
        self,
        features,
        cf,
        train,
        label="click",
        epsilon0=None,
        delta=None,
        removeNegativeValues=False,
        maxNbModalities=None,
    ):
        self.label = label
        self.featuresSet = FeaturesSet(features, "*&*", train, maxNbModalities)
        self.features = self.featuresSet.features
        self.aggClicks = self.featuresSet.Project(train, label)
        train["c"] = 1
        self.aggDisplays = self.featuresSet.Project(train, "c")
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
