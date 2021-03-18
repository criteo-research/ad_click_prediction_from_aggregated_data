import pandas as pd
import numpy as np
import random
import bisect
from collections import Counter
from agg_models.featuremappings import (
    CrossFeaturesMapping,
    FeatureMapping,
    DataProjection,
)

MAXMODALITIES = 1e7

# set of samples of 'x' used internally by AggMRFModel
class SampleRdd:
    def __init__(
        self,
        projections,
        nbSamples=None,
        decollapseGibbs=False,
        sampleFromPY0=False,
        maxNbRowsperGibbsUpdate=50,
        data=None
    ):
        self.projections = projections
        self.decollapseGibbs = decollapseGibbs
        self.sampleFromPY0 = sampleFromPY0
        self.features = [p.feature for p in projections]
        self.featurenames = [f.Name for f in self.features]
        self.Size = nbSamples
        self.data = data
        self.allcrossmods = False
        self.use_spark_rdd = True
        self.prediction = None

    def UpdateSampleWithGibbs(self, model):
        self.data = model.updateSamplesWithGibbsRdd(self.data)
        self.data.cache()
        self.data.localCheckpoint()
        
    def UpdateSampleWeights(self, model):
        rdd_xwmulambs = model.compute_rdd_expdotproducts(self.data)
        rdd_xwmulambs.cache()
        rdd_weighted = model.compute_weights(rdd_xwmulambs)
        rdd_weighted.cache()
        rdd_xwmulambs.unpersist()
        self.data = model.compute_enoclick_eclick_withweight(rdd_weighted)
        self.data.localCheckpoint()

    def PredictInternal(self, model):
        self.UpdateSampleWeights(model)
        pdisplays, z0_on_z = model.getPredictionsVectorRdd(self.data)
        predict = pdisplays*self.Size/z0_on_z
        self.prediction = predict
        
    def GetPrediction(self, model):
        return self.prediction