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
        rdd_sample_weights_with_expdotproducts = model.compute_rdd_expdotproducts(self.data)
        # rdd_sample_weights_with_expdotproducts.cache()
        rdd_sample_updated_weights_with_expdotproducts = model.compute_weights(rdd_sample_weights_with_expdotproducts)
        # rdd_sample_updated_weights_with_expdotproducts.cache()
        # rdd_sample_weights_with_expdotproducts.unpersist()
        self.data = model.compute_enoclick_eclick_zi(rdd_sample_updated_weights_with_expdotproducts)
        self.data.localCheckpoint()
        
    def compute_prediction(self, model):
        pdisplays, z_on_z0 = model.getPredictionsVectorRdd(self.data)
        # Compute z0_on_z : 1 / np.mean(z_i) = np.sum(z_zi) / nbSamples
        predict = pdisplays*self.Size/z_on_z0
        self.prediction = predict

    def PredictInternal(self, model):
        rdd_sample_updated_weights_with_expdotproducts = model.compute_rdd_expdotproducts(self.data)
        self.data = model.compute_enoclick_eclick_zi(rdd_sample_updated_weights_with_expdotproducts)
        self.data.localCheckpoint()
        self.compute_prediction(model)

    def GetPrediction(self, model):        
        return self.prediction
