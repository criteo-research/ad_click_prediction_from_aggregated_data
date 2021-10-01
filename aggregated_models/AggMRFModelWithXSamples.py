import numpy as np
import pandas as pd
import pyspark.sql as ps
import pyspark.sql.functions as F
from dataclasses import dataclass, asdict
from aggregated_models.featuremappings import SingleFeatureMapping, CrossFeaturesMapping
from aggregated_models.SampleSet import SampleSet, FullSampleSet
from aggregated_models.SampleRdd import SampleRdd
from aggregated_models import featureprojections
from aggregated_models.baseaggmodel import BaseAggModel, WeightsSet

import logging
from aggregated_models.aggdataset import AggDataset, FeaturesSet
from thx.hadoop.spark_config_builder import SparkSession
from aggregated_models.noiseDistributions import *

from aggregated_models.agg_mrf_model import *


class AggMRFModelWithXSamples(AggMRFModel):
    def __init__(
        self,
        aggdata: AggDataset,
        config_params: AggMRFModelParams,
        sparkSession: Optional[SparkSession] = None,
    ):
        super().__init__(aggdata, config_params, sparkSession)

    def defineSamples(self, XsamplesDF, nbpartitions=1000):
        features = self.features
        if self.sparkSession is None:
            self.Xsamples = self.transformDf(XsamplesDF)[features].values
        else:
            rdd = self.transformDf(XsamplesDF).rdd
            self.Xsamples = rdd.map(lambda x: [x[f] for f in features]).repartition(nbpartitions).cache()
            self.nbXsamples = self.Xsamples.count()
        self.resetSamples()

    def resetSamples(self):
        if self.sparkSession is None:
            self.samples.set_data_from_rows(self.Xsamples)
        else:
            self.samples.rddSamples = self.Xsamples
            self.samples.nbSamples = self.nbXsamples

    def updateSamplesWithGibbs(self, samples):
        self.resetSamples()
        if self.nbGibbsIter > 0:
            samples.rddSamples = samples._runGibbsSampling(self, self.nbGibbsIter)
        if self.sparkSession is not None:
            samples.cleanBroadcasts()
        samples.UpdateSampleWeights(self)
