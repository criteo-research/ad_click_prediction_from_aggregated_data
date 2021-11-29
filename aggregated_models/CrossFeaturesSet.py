from aggregated_models.RawFeatureMapping import *
from aggregated_models.FeatureEncodings import *

import pandas as pd
import pickle
import pyspark.sql.functions as F


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


def getMaxNbModalities(var, maxNbModalities):
    if type(maxNbModalities) is dict:
        if var in maxNbModalities:
            return maxNbModalities[var]
        else:
            return maxNbModalities["default"]
    return maxNbModalities


class CrossFeaturesSet:
    def __init__(self, rawFeaturesSet: RawFeaturesSet, crossfeaturesStr, maxNbModalities=None):

        self.rawFeaturesSet = rawFeaturesSet
        self.crossfeaturesStr = crossfeaturesStr
        self.maxNbModalities = maxNbModalities
        self.build()

    @property
    def features(self):
        return self.rawFeaturesSet.features

    def build(self):
        self.crossfeatures = parseCF(self.features, self.crossfeaturesStr)
        allfeatures = [f for cf in self.crossfeatures for f in cf]
        if any([f not in self.features for f in allfeatures]):
            raise Exception("Error: Some cross feature not declared in features list ")
        self.buildEncodings()

    def buildEncodings(self):
        self.encodings = {}
        for i, f in enumerate(self.features):
            maxNbModalities = getMaxNbModalities(f, self.maxNbModalities)
            rawMapping = self.rawFeaturesSet.rawmappings[f]
            self.encodings[f] = SingleFeatureEncoding.FromRawFeatureMapping(i, rawMapping, maxNbModalities)

        for cf in self.crossfeatures:
            if len(cf) != 2:
                raise Exception("cf of len !=2  not supported yet")
            maxNbModalities = getMaxNbModalities(GetCfName(cf), self.maxNbModalities)
            encoding = CrossFeatureEncoding.FromSingleFeatureEncodings(
                self.encodings[cf[0]], self.encodings[cf[1]], maxNbModalities
            )
            self.encodings[encoding.Name] = encoding

    def dump(self, handle):
        self.rawFeaturesSet.dump(handle)
        pickle.dump(self.crossfeaturesStr, handle)
        pickle.dump(self.maxNbModalities, handle)

    @staticmethod
    def load(handle):
        rawFeaturesSet = RawFeaturesSet.load(handle)
        crossfeaturesStr = pickle.load(handle)
        maxNbModalities = pickle.load(handle)
        return CrossFeaturesSet(rawFeaturesSet, crossfeaturesStr, maxNbModalities)

    def transformDf(self, df, alsoCrossfeatures=False):
        if alsoCrossfeatures:
            print("TODO :  reimplement 'alsoCrossfeatures' ")
            error
        return self.rawFeaturesSet.Map(df)

    def __repr__(self):
        return ",".join(f.Name for f in self.encodings.values())

    def fix_fids(self, features_sublist):
        # print("baseAggModel::fix_fids ")
        fid = 0
        encodings = self.encodings
        for f in features_sublist:
            mapping = encodings[f]
            mapping._fid = fid
            fid += 1

    @staticmethod
    def FromDf(df, features, maxNbModalities, crossfeaturesStr="*&*"):
        rawFeaturesSet = RawFeaturesSet.FromDF(features, df)
        return CrossFeaturesSet(rawFeaturesSet, crossfeaturesStr, maxNbModalities)
