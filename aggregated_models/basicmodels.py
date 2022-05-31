import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import FeatureHasher
import pandas as pd
#from aggregated_models.aggdataset import CrossFeaturesSet


# Wrapper around sklearn.feature_extraction.FeatureHasher
class Hasher:
    def __init__(self, features, hashspace=2 ** 16):
        self.features = features
        self.hashspace = hashspace

    def __str__(self):
        return "features = " + str(self.features) + "; hashspace =" + str(self.hashspace)

    def features_to_list_of_strings(self, row):
        return [f"{feature}_{row[feature]}" for feature in row.index]

    def hash_features(self, df):
        raw_features = df[self.features]
        features_as_list_of_strings = raw_features.apply(self.features_to_list_of_strings, axis=1)
        hasher = FeatureHasher(n_features=self.hashspace, input_type="string", alternate_sign=False)
        features = hasher.fit_transform(features_as_list_of_strings)
        return features


# simple logistic regression from hashed data. No crossfeatures.
# wrapper around sklearn.linear_model.LogisticRegression
class LogisticModel:
    def __init__(
        self,
        label,
        features,
        lambdaL2=1.0,
        hashspace=2 ** 16,
        max_iter=50,
    ):
        self.model = LogisticRegression(
                max_iter=max_iter, C=0.5 / lambdaL2
            )  # multiplying by 0.5 to get results similar to my own reimplem ??

        self.label = label
        self.features = features
        self.lambdaL2 = lambdaL2
        self.hasher = Hasher(features, hashspace)

    def fit(self, df):
        labels = df[self.label]
        featuresdf = self.hasher.hash_features(df)
        with warnings.catch_warnings(record=True):
            self.model.fit(featuresdf, labels)

    def predict_proba(self, df):
        featuresdf = self.hasher.hash_features(df)
        return self.model.predict_proba(featuresdf)[:, 1]

    def predictDF(self, df, pred_col_name: str):
        df = df.copy()
        df[pred_col_name] = self.predict_proba(df)
        return df


# Logistic regression with cross features (ie 2e order kernell)
class LogisticModelWithCF:
    def __init__(
        self,
        label,
        features,
        crossfeatures,
        train,
        lambdaL2=1.0,
        hashspace=2 ** 16,
        max_iter=50,
    ):
        self.featuresSet = FeaturesSet(features, crossfeatures, train)
        self.features = [x for x in self.featuresSet.mappings]
        self.model = LogisticModel(label, self.features, lambdaL2, hashspace, max_iter)
        self.lambdaL2 = lambdaL2

    def fit(self, df):
        dfWithCfs = self.featuresSet.transformDf(df, True)
        self.model.fit(dfWithCfs)

    def predict_proba(self, df):
        dfWithCfs = self.featuresSet.transformDf(df, True)
        return self.model.predict_proba(dfWithCfs)

    def predictDF(self, df, pred_col_name: str):
        dfWithCfs = self.featuresSet.transformDf(df, True)
        return self.model.predictDF(dfWithCfs, pred_col_name)

    def computeLossAndRegul(self, train):
        p = logisticCfs.predict_proba(train)
        y = train.click.values
        llh = -sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        p0 = sum(y) / len(train)
        llh -= -sum(y * np.log(p0) + (1 - y) * np.log(1 - p0))
        w = a = logisticCfs.model.model.coef_[0]
        regul = w.dot(w) * 200 / self.nbCoefs
        return np.array([llh + regul, llh, regul])


class NaiveBayesModel:
    def __init__(self, label, features, lambdaL2=1.0):
        self.label = label
        self.features = features
        self.lambdaL2 = lambdaL2

    def fit(self, df):
        labels = df[self.label]
        self.models = {}
        for var in self.features:
            self.models[var] = LogisticModel(self.label, features=[var], lambdaL2=self.lambdaL2)
            self.models[var].fit(df)

    def predict_proba(self, df):
        y = df[self.label].values
        py = y.sum() / len(df)
        unnormalizedP0 = y * 0 + 1 - py
        unnormalizedP1 = y * 0 + py
        for var in self.features:
            predictions = self.models[var].predict_proba(df)
            unnormalizedP1 = unnormalizedP1 * predictions / py
            unnormalizedP0 = unnormalizedP0 * (1 - predictions) / (1 - py)

        return unnormalizedP1 * 1.0 / (unnormalizedP1 + unnormalizedP0)

    def predictDF(self, df, pred_col_name: str):
        df = df.copy()
        df[pred_col_name] = self.predict_proba(df)
        return df

    def computeLLH(self, df):
        y = df[self.label].values
        predictions = self.predict_proba(df)
        return LlhCVN(predictions, y)

    
    
class FeatureMapping():
    """class representing one feature and its set of modalities.
    """
    def __init__ (self, name:str, train:pd.DataFrame, fid:int = 0):
        """var: name of the feature 
           df:  pd.DataFrame containing this feature
           fid:  index of the feature in np.arrays of shape ( nbfeatures,nbsamples )"""
        self.Name = name
        self._modalities = np.array( sorted(set(train[name].values))) # list of modalities observed in train df
        self._dicoModalityToId = { m: i  for i, m in enumerate( self._modalities )} # assigning an id to each modality
        self._default = len(self._modalities) # assigning an id for eventual modalities for observed in train 
        self.Size =  len( self._modalities ) +1   # +1 To get a modality for "unobserved"
        self._fid = fid

    # replace initial modalities of features by modality index
    def Map( self , df):
        df[self.Name] = df[self.Name].apply( lambda x:  self._dicoModalityToId.get( x , self._default ) )
        return df
    # inverse transorm of Map
    def RetrieveModalities(self,df):
        df[self.Name] = df[self.Name].apply( lambda x: -1 if x ==self._default else self._modalities[x] )
        return df
        
   # returning feature values from a np.array of shape (nbfeatures, nbsamples) 
    def Values_(self, x):
        return x[ self._fid ]

    # x : np.array of shape (nbfeatures, nbsamples) 
    # y : array of len( nbsamples )
    # return array with, for each modality m of feature: sum( y on samples where feature modality is m)  
    def Project_( self, x , y  ):
        x = x[ self._fid ]
        return projectNUMBA( x, y , self.Size)

    def Values(self, df :pd.DataFrame ):
        return df[ self.Name ].values

    # df : dataframe 
    # col : column of the df to sum
    # return array with, for each modality m of feature: sum( y on rows where feature modality is m)      
    def Project( self, df: pd.DataFrame , col  ):
        groupedDF = df[[ self.Name, col]].groupby( self.Name ).sum()
        data = np.zeros(self.Size)
        data[ groupedDF.index ] = groupedDF[col]
        return data

    def __repr__(self):
        return  f"{self.Name}({self.Size})"

    def toDF(self):
        return pd.DataFrame( { self.Name: np.arange(0,self.Size) } )

    
class CrossFeaturesMapping():
    """ a crossfeature between two single features.
    """
    def __init__ (self, singleFeature1 : FeatureMapping,
                        singleFeature2 : FeatureMapping ):
        self._variables = [ singleFeature1, singleFeature2 ]
        self._v1 = self._variables[0].Name
        self._v2 = self._variables[1].Name
        self._fid1 = self._variables[0]._fid
        self._fid2 = self._variables[1]._fid
        self.Name = GetCfName([ f.Name for f in self._variables])
        self.coefV2 = singleFeature1.Size
        self.Size = singleFeature1.Size * singleFeature2.Size

    def Values_(self, x):
        return x[ self._fid1 ] + self.coefV2 * x[ self._fid2 ]
    def Project_( self, x , y  ):
        x = self.Values_(x)
        return projectNUMBA( x, y , self.Size)
    
    def Values(self, df):
        return df[ self._v1 ].values + self.coefV2 * df[ self._v2 ].values    
    def Project( self, df , col  ):
        data = np.zeros(self.Size)

        groupedDF = df[[ self._v1,self._v2, col]].groupby([ self._v1,self._v2] ).sum().reset_index()
        data[ self.Values(groupedDF) ] = groupedDF[col]
        return data        
        
    def Map( self , df):
        df[self.Name] = df[ self._v1 ] + self.coefV2 * df[ self._v2 ]
        return df
    def __repr__(self):
        return "x".join( [str(x) for x in self._variables ])

    def ValueVar1(self,x):
        return x % self.coefV2
    def ValueVar2(self,x):
        return (x - x % self.coefV2) / self.coefV2
    
    def toDF(self):
        df = pd.DataFrame(  { self.Name: np.arange(0, self.Size)} )
        df[ self._v1 ] = self.ValueVar1( df[self.Name]) 
        df[ self._v2 ] = self.ValueVar2( df[self.Name]).astype(int)  
        return df
    
def GetCfName(variables):   
    return "&".join( sorted(variables))

def parseCFNames(features,crossfeaturesStr):
    cfs = parseCF(features,crossfeaturesStr)
    return [  GetCfName(cf) for cf in cfs]

def parseCF(features,crossfeaturesStr):
    cfs = []
    crossfeaturesStr = crossfeaturesStr.split( "|" )
    for cfStr in crossfeaturesStr:
        cfFeatures = cfStr.split("&")
        nbWildcards = len(  [ f for f in cfFeatures if  f == "*" ])
        cfFeatures = [[ f for f in cfFeatures if not f == "*" ]]
        for i in range( 0, nbWildcards ):
            cfFeatures = [ cf + [v]  for v in features for cf in cfFeatures ]
        cfFeatures = [ sorted(f) for f in cfFeatures ] 
        cfs += cfFeatures
    cfs = [ list(sorted(set(cf))) for cf in cfs ]
    cfs = [ cf for cf in cfs if len(cf)==2 ]
    #remove duplicates
    dicoCfsStr = {}
    for cf in cfs:
        s = "&".join( [str(f) for f in cf] )
        dicoCfsStr[s] = cf
    cfs = [ cf for cf in dicoCfsStr.values()]
    return cfs    
    
class FeaturesSet():
    def __init__(self , features, crossfeaturesStr, df):
        self.features = features
        self.crossfeatures = parseCF(features, crossfeaturesStr)
        
        allfeatures = [ f for cf in self.crossfeatures for f in cf  ]
        if any( [ f not in features for f in allfeatures ]) :
               raise  Exception('Error: Some cross feature not declared in features list ')
        self.buildFeaturesMapping(df)
 
    def buildFeaturesMapping(self , df):
        mappings = {}
        for var in self.features:
            mappings[var] = FeatureMapping(var, df )
        fid = 0
        for var in self.features:
            mappings[var] = FeatureMapping(var, df , fid )
            fid += 1
    
        for cf in self.crossfeatures:
            if len(cf) != 2:
                raise Exception( "cf of len !=2  not supported yet")
            mapping = CrossFeaturesMapping( mappings[cf[0]],mappings[cf[1]] )
            mappings[ mapping.Name ] = mapping
        self.mappings = mappings
    
    def getMapping(self,var):
        return self.mappings[var]
    
    def transformDf( self, df , alsoCrossfeatures = True ):
        df = df.copy()
        for var in self.mappings.values():
            if alsoCrossfeatures or type(var) is FeatureMapping:
                df = var.Map(df)
        return df
    
    def Project(self, train, column ):
        train = self.transformDf(train)   
        projections = {}
        for var in self.mappings:
            projections[ var ] = DataProjection( self.mappings[var] , train , column )
        return projections    
    
    def __repr__(self):
        return ",".join( f.Name for f in self.mappings.values() )
    