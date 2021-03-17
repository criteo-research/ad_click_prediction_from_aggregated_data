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
        samplesRdd=None
    ):
        self.projections = projections
        self.decollapseGibbs = decollapseGibbs
        self.sampleFromPY0 = sampleFromPY0
        self.features = [p.feature for p in projections]
        self.featurenames = [f.Name for f in self.features]
        self.Size = nbSamples
        self.samplesRdd = samplesRdd
        self.use_spark_rdd = True

    def UpdateSampleWithGibbs(self, model):
        # Cannot cache + unpersist when checkpointing
        # previousRdd = self.samplesRdd
        self.samplesRdd = model.updateSamplesWithGibbsRdd(self.samplesRdd)
        self.samplesRdd.cache()
        self.samplesRdd.localCheckpoint()
        # Unpersist yields error after checkpoint because previousRdd has been GCed
        # previousRdd.unpersist()
        rdd_xwmulambs = model.compute_rdd_expdotproducts(self.samplesRdd)
        rdd_xwmulambs.cache()
        rdd_p_display = model.compute_enoclick_eclick(rdd_xwmulambs)
        rdd_p_display.cache()
        rdd_xwmulambs.unpersist()
        self.predictions, z0 = model.getPredictionsVectorRdd(rdd_p_display)
        self.predictions /= z0
        rdd_p_display.unpersist()

    def Predict(self, model):        
        rdd_xwexpmulambs = model.compute_rdd_expdotproducts(self.samplesRdd)
        rdd_xwexpmulambs.cache()
        rdd_p_display = model.compute_enoclick_eclick_withweight(rdd_xwexpmulambs)
        rdd_p_display.cache()
        rdd_xwexpmulambs.unpersist()
        pdisplays, z0_on_z = model.getPredictionsVectorRdd(rdd_p_display)
        self.pdisplays = pdisplays/z0_on_z
        rdd_p_display.unpersist()